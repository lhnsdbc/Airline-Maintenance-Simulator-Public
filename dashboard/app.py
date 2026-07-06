"""Dash policy-comparison dashboard for synthetic experiment artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import plotly.express as px
from dash import Dash, Input, Output, State, dash_table, dcc, html


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
            html.Div(className="metric-rule"),
            html.Div(title, className="metric-title"),
            html.Div(value, className="metric-value"),
            html.Div(detail, className="metric-detail"),
        ],
        className="metric-card",
    )


def _comparison_options(comparisons: list[str]) -> list[dict[str, str]]:
    return [{"label": item, "value": item} for item in comparisons]


def _add_marker_size(current: pd.DataFrame) -> pd.DataFrame:
    current = current.copy()
    min_score = current["policy_quality_score"].min()
    current["marker_size_score"] = (current["policy_quality_score"] - min_score + 1).clip(lower=1)
    return current


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
                            html.Div("Synthetic Portfolio Evidence", className="eyebrow"),
                            html.H1("Maintenance Policy Evaluation"),
                            html.P("Policy rungs, non-routine workload signals, and tracked KPI artifacts."),
                            html.Div(id="refresh-status", className="refresh-status"),
                        ],
                        className="title-block",
                    ),
                    html.Div(
                        [
                            html.Label("Experiment", htmlFor="comparison-select"),
                            dcc.Dropdown(
                                id="comparison-select",
                                options=_comparison_options(comparisons),
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
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H2("Policy Ladder"),
                                    html.P("Compare the selected KPI across rungs and NR modes."),
                                ],
                                className="panel-heading",
                            ),
                            dcc.Graph(id="policy-chart"),
                        ],
                        className="panel wide",
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H2("Reserve Alignment"),
                                    html.P("Reserved vs realized NR labour by run."),
                                ],
                                className="panel-heading",
                            ),
                            dcc.Graph(id="nr-chart"),
                        ],
                        className="panel",
                    ),
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
            dcc.Interval(id="artifact-refresh", interval=5000, n_intervals=0),
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
                :root {
                    --ink: #172033;
                    --muted: #65758b;
                    --paper: #fbfcfe;
                    --panel: #ffffff;
                    --line: #d7e0ea;
                    --navy: #102033;
                    --teal: #0f766e;
                    --amber: #b45309;
                    --steel: #64748b;
                }
                body {
                    margin: 0;
                    background:
                        linear-gradient(90deg, rgba(16, 32, 51, 0.035) 1px, transparent 1px),
                        linear-gradient(180deg, rgba(16, 32, 51, 0.03) 1px, transparent 1px),
                        var(--paper);
                    background-size: 32px 32px;
                    color: var(--ink);
                    font-family: "Aptos", "Segoe UI", sans-serif;
                }
                .app-shell { padding: 28px; max-width: 1480px; margin: 0 auto; }
                .topbar {
                    display: grid;
                    grid-template-columns: minmax(360px, 1fr) minmax(240px, 320px) minmax(220px, 300px);
                    gap: 16px;
                    align-items: end;
                    margin-bottom: 20px;
                    padding-bottom: 18px;
                    border-bottom: 2px solid var(--navy);
                }
                .eyebrow {
                    display: inline-block;
                    color: var(--teal);
                    font-size: 12px;
                    font-weight: 800;
                    letter-spacing: 0.08em;
                    text-transform: uppercase;
                    margin-bottom: 8px;
                }
                .title-block h1 { margin: 0; font-size: 34px; line-height: 1.05; letter-spacing: 0; }
                .title-block p { margin: 8px 0 0; color: var(--muted); max-width: 760px; }
                .refresh-status { margin-top: 8px; min-height: 18px; color: var(--steel); font-size: 12px; }
                .control label {
                    display: block;
                    font-size: 12px;
                    font-weight: 700;
                    text-transform: uppercase;
                    color: var(--muted);
                    margin-bottom: 6px;
                }
                .control .Select-control {
                    border-color: var(--line);
                    border-radius: 8px;
                    min-height: 42px;
                }
                .metric-strip {
                    display: grid;
                    grid-template-columns: repeat(4, minmax(0, 1fr));
                    gap: 14px;
                    margin-bottom: 16px;
                }
                .metric-card, .panel {
                    background: var(--panel);
                    border: 1px solid var(--line);
                    border-radius: 8px;
                    box-shadow: 0 12px 30px rgba(16, 32, 51, 0.07);
                }
                .metric-card {
                    position: relative;
                    padding: 16px 18px 15px;
                    min-height: 96px;
                    overflow: hidden;
                }
                .metric-rule {
                    position: absolute;
                    left: 0;
                    top: 0;
                    bottom: 0;
                    width: 5px;
                    background: linear-gradient(180deg, var(--teal), var(--amber));
                }
                .metric-title {
                    font-size: 12px;
                    font-weight: 700;
                    color: var(--muted);
                    text-transform: uppercase;
                }
                .metric-value { font-size: 30px; font-weight: 780; margin-top: 8px; line-height: 1.05; }
                .metric-detail {
                    color: var(--muted);
                    font-size: 13px;
                    margin-top: 5px;
                    overflow-wrap: anywhere;
                }
                .grid { display: grid; grid-template-columns: 1.35fr 1fr; gap: 14px; }
                .panel { padding: 12px; min-width: 0; }
                .panel-heading {
                    display: flex;
                    align-items: baseline;
                    justify-content: space-between;
                    gap: 12px;
                    padding: 2px 8px 0;
                }
                .panel-heading h2 {
                    margin: 0;
                    font-size: 16px;
                    line-height: 1.2;
                }
                .panel-heading p {
                    margin: 0;
                    color: var(--muted);
                    font-size: 13px;
                    text-align: right;
                }
                .table-panel { margin-top: 14px; padding: 16px; }
                .table-panel h2 { margin: 0 0 12px; font-size: 18px; }
                @media (max-width: 980px) {
                    .app-shell { padding: 16px; }
                    .topbar, .grid, .metric-strip { grid-template-columns: 1fr; }
                    .panel-heading { display: block; }
                    .panel-heading p { text-align: left; margin-top: 4px; }
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
        Output("comparison-data", "data"),
        Output("comparison-select", "options"),
        Output("comparison-select", "value"),
        Output("refresh-status", "children"),
        Input("artifact-refresh", "n_intervals"),
        State("comparison-select", "value"),
    )
    def refresh_artifacts(_n_intervals, selected_comparison):
        fresh = load_comparisons(artifact_dir)
        if fresh.empty:
            return [], [], None, "No experiment artifacts loaded yet."

        fresh_comparisons = sorted(fresh["comparison_id"].unique())
        comparison_id = selected_comparison if selected_comparison in fresh_comparisons else fresh_comparisons[-1]
        status = f"Loaded {len(fresh)} run rows from {len(fresh_comparisons)} comparison set(s)."
        return fresh.to_dict("records"), _comparison_options(fresh_comparisons), comparison_id, status

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
        current = _add_marker_size(current)

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
            size="marker_size_score",
            color="rung",
            symbol="nr_mode",
            hover_data=["run_id", "policy_quality_score", "nr_uncovered_labor_hours"],
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
