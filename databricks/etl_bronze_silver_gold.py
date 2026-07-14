"""Databricks-compatible counterpart to the scheduled synthetic ETL pipeline.

Attach this file to a Databricks job when Spark is available. It deliberately uses
the same public synthetic schemas as ``pipeline.run`` and writes Delta tables.
"""

from pyspark.sql import functions as F


def run(spark, source_path: str, target_path: str) -> None:
    aircraft = spark.read.option("header", True).csv(f"{source_path}/AircraftRegistrations.csv")
    airports = spark.read.option("header", True).csv(f"{source_path}/Airports.csv")

    aircraft.dropDuplicates(["AircraftRegistrationFull"]).write.format("delta").mode("overwrite").save(
        f"{target_path}/bronze/aircraft"
    )
    airports.dropDuplicates(["IataAirportCode"]).write.format("delta").mode("overwrite").save(
        f"{target_path}/bronze/airports"
    )

    silver_aircraft = aircraft.dropna(subset=["AircraftRegistrationFull"]).dropDuplicates(["AircraftRegistrationFull"])
    silver_aircraft.write.format("delta").mode("overwrite").save(f"{target_path}/silver/aircraft")

    gold_fleet = silver_aircraft.groupBy("AircraftTypeCodeIATA").agg(
        F.count("AircraftRegistrationFull").alias("aircraft_count")
    )
    gold_fleet.write.format("delta").mode("overwrite").save(f"{target_path}/gold/fleet_by_type")
