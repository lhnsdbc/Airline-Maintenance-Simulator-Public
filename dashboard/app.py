"""Dash policy-comparison dashboard for synthetic experiment artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import plotly.express as px
from dash import Dash, Input, Output, dash_table, dcc, html


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_DIR = REPO_ROOT / "artifacts" / "experiments"

METRIC_OPTIONS = [
    ("Policy quality", "policy_quality_score"),
    ("Departure delay", "flights_delay_dep"),
    ("Interval spillage", "interval_spillage"),
    ("Uncovered NR labour", "nr_uncovered_labor_hours"),
    ("Completion factor", "completion_factor"),
    ("Reserve-realized correlation", "nr_reserve_realized_corr"),
]


def load_comparisons(artifact_dir: Path = DEFAULT_ARTIFACT_DIR) -> pd.DataFrame:
    frames = []
    for path in sorted(artifact_dir.glob("*_comparison_seed*/kpis.csv")):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        df["comparison_id"] = path.parent.name
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _empty_figure(message: str):
    fig = px.scatter()
    fig.update_layout(
        annotations=[{
            "text": message,
            "xref": "paper",
            "yref": "paper",
            "x": 0.5,
            "y": 0.5,
            "showarrow": False,
        }],
        template="plotly_white",
        height=360,
    )
    return fig


def _metric_card(title: str, value: str, detail: str):
    return html.Div(
        [
            html.Div(title, className="metric-title"),
            html.Div(value, className="metric-value"),
            html.Div(detail, className="metric-detail"),
        ],
        className="metric-card",
    )


def create_app(artifact_dir: Path = DEFAULT_ARTIFACT_DIR) -> Dash:
    app = Dash(__name__)
    df = load_comparisons(artifact_dir)
    comparisons = sorted(df["comparison_id"].unique()) if not df.empty else []
    default_comparison = comparisons[-1] if comparisons else None

    app.layout = html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.H1("Maintenance Policy Evaluation"),
                            html.P("Synthetic experiment tracking dashboard"),
                        ],
                        className="title-block",
                    ),
                    html.Div(
                        [
                            html.Label("Experiment", htmlFor="comparison-select"),
                            dcc.Dropdown(
                                id="comparison-select",
                                options=[{"label": item, "value": item} for item in comparisons],
                                value=default_comparison,
                                clearable=False,
                            ),
                        ],
                        className="control",
                    ),
                    html.Div(
                        [
                            html.Label("Primary KPI", htmlFor="metric-select"),
                            dcc.Dropdown(
                                id="metric-select",
                                options=[{"label": label, "value": value} for label, value in METRIC_OPTIONS],
                                value="policy_quality_score",
                                clearable=False,
                            ),
                        ],
                        className="control",
                    ),
                ],
                className="topbar",
            ),
            html.Div(id="metric-strip", className="metric-strip"),
            html.Div(
                [
                    html.Div(dcc.Graph(id="policy-chart"), className="panel wide"),
                    html.Div(dcc.Graph(id="nr-chart"), className="panel"),
                ],
                className="grid",
            ),
            html.Div(
                [
                    html.H2("Run Comparison"),
                    dash_table.DataTable(
                        id="kpi-table",
                        page_size=8,
                        sort_action="native",
                        style_as_list_view=True,
                        style_cell={
                            "fontFamily": "Segoe UI, sans-serif",
                            "fontSize": "13px",
                            "padding": "8px",
                            "textAlign": "left",
                            "whiteSpace": "normal",
                            "height": "auto",
                        },
                        style_header={
                            "backgroundColor": "#0f172a",
                            "color": "#f8fafc",
                            "fontWeight": "600",
                            "border": "0",
                        },
                        style_data={"border": "0", "borderBottom": "1px solid #e2e8f0"},
                    ),
                ],
                className="panel table-panel",
            ),
            dcc.Store(id="comparison-data", data=df.to_dict("records")),
        ],
        className="app-shell",
    )

    app.index_string = """
    <!DOCTYPE html>
    <html>
        <head>
            {%metas%}
            <title>Maintenance Policy Evaluation</title>
            {%favicon%}
            {%css%}
            <style>
                body {
                    margin: 0;
                    background: #eef2f7;
                    color: #172033;
                    font-family: "Segoe UI", "Aptos", sans-serif;
                }
                .app-shell { padding: 24px; max-width: 1440px; margin: 0 auto; }
                .topbar {
                    display: grid;
                    grid-template-columns: minmax(280px, 1fr) minmax(240px, 320px) minmax(220px, 300px);
                    gap: 16px;
                    align-items: end;
                    margin-bottom: 18px;
                }
                .title-block h1 { margin: 0; font-size: 28px; letter-spacing: 0; }
                .title-block p { margin: 6px 0 0; color: #475569; }
                .control label {
                    display: block;
                    font-size: 12px;
                    font-weight: 700;
                    text-transform: uppercase;
                    color: #475569;
                    margin-bottom: 6px;
                }
                .metric-strip {
                    display: grid;
                    grid-template-columns: repeat(4, minmax(0, 1fr));
                    gap: 12px;
                    margin-bottom: 14px;
                }
                .metric-card, .panel {
                    background: #ffffff;
                    border: 1px solid #d9e2ef;
                    border-radius: 8px;
                    box-shadow: 0 10px 28px rgba(15, 23, 42, 0.06);
                }
                .metric-card { padding: 14px 16px; min-height: 88px; }
                .metric-title {
                    font-size: 12px;
                    font-weight: 700;
                    color: #64748b;
                    text-transform: uppercase;
                }
                .metric-value { font-size: 28px; font-weight: 750; margin-top: 6px; }
                .metric-detail { color: #64748b; font-size: 13px; margin-top: 2px; }
                .grid { display: grid; grid-template-columns: 1.35fr 1fr; gap: 14px; }
                .panel { padding: 10px; min-width: 0; }
                .table-panel { margin-top: 14px; padding: 16px; }
                .table-panel h2 { margin: 0 0 12px; font-size: 18px; }
                @media (max-width: 980px) {
                    .app-shell { padding: 16px; }
                    .topbar, .grid, .metric-strip { grid-template-columns: 1fr; }
                }
            </style>
        </head>
        <body>
            {%app_entry%}
            <footer>{%config%}{%scripts%}{%renderer%}</footer>
        </body>
    </html>
    """

    @app.callback(
        Output("metric-strip", "children"),
        Output("policy-chart", "figure"),
        Output("nr-chart", "figure"),
        Output("kpi-table", "data"),
        Output("kpi-table", "columns"),
        Input("comparison-select", "value"),
        Input("metric-select", "value"),
        Input("comparison-data", "data"),
    )
    def update_dashboard(comparison_id, metric_name, records):
        current = pd.DataFrame(records)
        if current.empty or comparison_id is None:
            fig = _empty_figure("Run `py -m experiments.synthetic_experiment` to create artifacts.")
            return [], fig, fig, [], []

        current = current[current["comparison_id"] == comparison_id].copy()
        if current.empty:
            fig = _empty_figure("No rows found for selected comparison.")
            return [], fig, fig, [], []

        best = current.sort_values("policy_quality_score", ascending=False).iloc[0]
        delay_best = current.sort_values("flights_delay_dep", ascending=True).iloc[0]
        uncovered_best = current.sort_values("nr_uncovered_labor_hours", ascending=True).iloc[0]
        cards = [
            _metric_card("Best policy", f"{best['rung']} / {best['nr_mode']}", f"Score {best['policy_quality_score']:.1f}"),
            _metric_card("Lowest delay", f"{delay_best['flights_delay_dep']:.0f}", f"{delay_best['rung']} / {delay_best['nr_mode']}"),
            _metric_card("Lowest uncovered NR", f"{uncovered_best['nr_uncovered_labor_hours']:.2f} h", f"{uncovered_best['rung']} / {uncovered_best['nr_mode']}"),
            _metric_card("Runs compared", str(len(current)), comparison_id),
        ]

        chart = px.bar(
            current,
            x="rung",
            y=metric_name,
            color="nr_mode",
            barmode="group",
            text_auto=".2s",
            color_discrete_map={"static": "#64748b", "predicted": "#0f766e"},
            template="plotly_white",
        )
        chart.update_layout(height=390, margin={"l": 42, "r": 20, "t": 28, "b": 42}, legend_title_text="")

        nr_chart = px.scatter(
            current,
            x="nr_reserved_hours",
            y="nr_realized_hours",
            size="policy_quality_score",
            color="rung",
            symbol="nr_mode",
            hover_data=["run_id", "nr_uncovered_labor_hours"],
            template="plotly_white",
        )
        max_nr = max(current["nr_reserved_hours"].max(), current["nr_realized_hours"].max())
        nr_chart.add_shape(type="line", x0=0, y0=0, x1=max_nr, y1=max_nr, line={"dash": "dot", "color": "#94a3b8"})
        nr_chart.update_layout(height=390, margin={"l": 42, "r": 20, "t": 28, "b": 42}, legend_title_text="")

        table_cols = [
            "run_id",
            "rung",
            "nr_mode",
            "policy_quality_score",
            "flights_delay_dep",
            "interval_spillage",
            "nr_uncovered_labor_hours",
            "completion_factor",
        ]
        table = current[table_cols].round(3)
        columns = [{"name": col.replace("_", " ").title(), "id": col} for col in table.columns]
        return cards, chart, nr_chart, table.to_dict("records"), columns

    return app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the policy-comparison dashboard.")
    parser.add_argument("--artifact-dir", default=str(DEFAULT_ARTIFACT_DIR))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--debug", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    app = create_app(Path(args.artifact_dir))
    app.run_server(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
