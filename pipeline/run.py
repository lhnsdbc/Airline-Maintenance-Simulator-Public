"""Bronze/Silver/Gold ETL pipeline for public synthetic simulator data."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class LakeWriter:
    root: Path

    def write_bytes(self, relative_path: str, content: bytes) -> str:
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return relative_path


class AzureLakeWriter(LakeWriter):
    def __init__(self, connection_string: str, file_system: str, root: Path) -> None:
        super().__init__(root)
        from azure.storage.blob import BlobServiceClient

        self.container = BlobServiceClient.from_connection_string(connection_string).get_container_client(file_system)
        try:
            self.container.create_container()
        except Exception:
            pass

    def write_bytes(self, relative_path: str, content: bytes) -> str:
        super().write_bytes(relative_path, content)
        self.container.upload_blob(relative_path, content, overwrite=True)
        return relative_path


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8")


def _check(frame: pd.DataFrame, name: str, key: str, required: list[str]) -> Dict[str, Any]:
    missing_columns = [column for column in required if column not in frame.columns]
    null_key_count = int(frame[key].isna().sum()) if key in frame.columns else len(frame)
    duplicate_key_count = int(frame[key].duplicated().sum()) if key in frame.columns else len(frame)
    return {
        "dataset": name,
        "rows": int(len(frame)),
        "missing_columns": missing_columns,
        "null_key_count": null_key_count,
        "duplicate_key_count": duplicate_key_count,
        "passed": not missing_columns and null_key_count == 0 and duplicate_key_count == 0,
    }


def _read_inputs(data_dir: Path, artifact_dir: Path) -> Dict[str, pd.DataFrame]:
    input_dir = data_dir / "input"
    with (input_dir / "schedules" / "schedule_2023-01-01_1weeks").open("rb") as handle:
        schedule = pickle.load(handle)
    comparison = sorted(artifact_dir.glob("*_comparison_seed*/kpis.csv"))
    return {
        "aircraft": pd.read_csv(input_dir / "AircraftRegistrations.csv"),
        "airports": pd.read_csv(input_dir / "Airports.csv"),
        "schedule": schedule,
        "kpis": pd.read_csv(comparison[-1]) if comparison else pd.DataFrame(),
    }


def _record_metadata(status: Dict[str, Any], output_root: Path) -> None:
    server = os.getenv("PIPELINE_SQL_SERVER")
    password = os.getenv("PIPELINE_SQL_PASSWORD")
    if not server or not password:
        path = output_root / "pipeline_runs.json"
        existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
        existing.append(status)
        path.write_text(json.dumps(existing[-50:], indent=2), encoding="utf-8")
        return

    import pymssql

    connection = pymssql.connect(
        server=server,
        user=os.getenv("PIPELINE_SQL_USERNAME", "simulatoradmin"),
        password=password,
        database=os.getenv("PIPELINE_SQL_DATABASE", "simulator"),
        login_timeout=30,
        timeout=30,
        tds_version="7.4",
    )
    cursor = connection.cursor()
    cursor.execute(
        "IF OBJECT_ID('dbo.pipeline_runs', 'U') IS NULL "
        "CREATE TABLE dbo.pipeline_runs (run_id VARCHAR(36) PRIMARY KEY, status VARCHAR(20), "
        "started_at DATETIME2, completed_at DATETIME2, bronze_rows INT, silver_rows INT, gold_rows INT, "
        "quality_passed BIT, details NVARCHAR(MAX))"
    )
    cursor.execute(
        "INSERT INTO dbo.pipeline_runs VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
        (
            status["run_id"], status["status"], status["started_at"], status["completed_at"],
            status["bronze_rows"], status["silver_rows"], status["gold_rows"],
            status["quality_passed"], json.dumps(status),
        ),
    )
    connection.commit()
    connection.close()


def run_pipeline(data_dir: Path, artifact_dir: Path, output_root: Path, writer: LakeWriter | None = None) -> Dict[str, Any]:
    writer = writer or LakeWriter(output_root)
    run_id = str(uuid.uuid4())
    started = datetime.now(timezone.utc)
    inputs = _read_inputs(data_dir, artifact_dir)
    quality = [
        _check(inputs["aircraft"], "aircraft", "AircraftRegistrationFull", ["AircraftRegistrationFull", "AircraftTypeCodeIATA"]),
        _check(inputs["airports"], "airports", "IataAirportCode", ["IataAirportCode", "IcaoAirportCode"]),
        _check(inputs["schedule"], "schedule", "FlightLegId", ["FlightLegId", "DepartureAirport", "ArrivalAirport", "ActualBlockTimeDuration"]),
    ]
    if (inputs["schedule"]["ActualBlockTimeDuration"] <= 0).any():
        quality.append({"dataset": "schedule_duration", "passed": False, "reason": "non-positive block duration"})

    date_partition = started.strftime("%Y-%m-%d")
    for name, frame in inputs.items():
        writer.write_bytes(f"bronze/{name}/ingest_date={date_partition}/{name}.csv", _csv_bytes(frame))

    aircraft = inputs["aircraft"].dropna(subset=["AircraftRegistrationFull"]).drop_duplicates("AircraftRegistrationFull")
    airports = inputs["airports"].dropna(subset=["IataAirportCode"]).drop_duplicates("IataAirportCode")
    schedule = inputs["schedule"].dropna(subset=["FlightLegId", "DepartureAirport", "ArrivalAirport"]).drop_duplicates("FlightLegId").copy()
    schedule["departure_weekday"] = schedule["ScheduledDepartureTimeAtHovUtcWeekday"].astype(int)
    schedule["block_hours"] = (schedule["ActualBlockTimeDuration"] / 60.0).round(3)
    for name, frame in {"aircraft": aircraft, "airports": airports, "schedule": schedule}.items():
        writer.write_bytes(f"silver/{name}/ingest_date={date_partition}/{name}.csv", _csv_bytes(frame))

    gold_operations = (
        schedule.groupby("departure_weekday", as_index=False)
        .agg(flight_legs=("FlightLegId", "count"), total_block_hours=("block_hours", "sum"), mean_block_hours=("block_hours", "mean"))
        .round(3)
    )
    writer.write_bytes(f"gold/operations_by_weekday/ingest_date={date_partition}/operations_by_weekday.csv", _csv_bytes(gold_operations))
    if not inputs["kpis"].empty:
        writer.write_bytes(f"gold/policy_kpis/ingest_date={date_partition}/policy_kpis.csv", _csv_bytes(inputs["kpis"]))

    completed = datetime.now(timezone.utc)
    status = {
        "run_id": run_id,
        "status": "succeeded" if all(item.get("passed", False) for item in quality) else "failed_quality",
        "started_at": started.isoformat(),
        "completed_at": completed.isoformat(),
        "bronze_rows": sum(len(frame) for frame in inputs.values()),
        "silver_rows": len(aircraft) + len(airports) + len(schedule),
        "gold_rows": len(gold_operations) + len(inputs["kpis"]),
        "quality_passed": all(item.get("passed", False) for item in quality),
        "quality_checks": quality,
        "lake_layers": ["bronze", "silver", "gold"],
    }
    writer.write_bytes("gold/pipeline_status/latest.json", json.dumps(status, indent=2).encode("utf-8"))
    _record_metadata(status, output_root)
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the synthetic Bronze/Silver/Gold pipeline.")
    parser.add_argument("--data-dir", default=str(REPO_ROOT / "Data"))
    parser.add_argument("--artifact-dir", default=str(REPO_ROOT / "artifacts" / "experiments"))
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "artifacts" / "data_lake"))
    args = parser.parse_args()
    output_root = Path(args.output_dir)
    connection_string = os.getenv("PIPELINE_STORAGE_CONNECTION_STRING")
    writer: LakeWriter = AzureLakeWriter(connection_string, os.getenv("PIPELINE_FILE_SYSTEM", "maintenance-lake"), output_root) if connection_string else LakeWriter(output_root)
    print(json.dumps(run_pipeline(Path(args.data_dir), Path(args.artifact_dir), output_root, writer), indent=2))


if __name__ == "__main__":
    main()
