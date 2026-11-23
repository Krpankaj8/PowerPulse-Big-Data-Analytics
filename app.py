"""PowerPulse interactive energy analytics dashboard."""

import json
import math
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from kafka import KafkaConsumer
from kafka.errors import KafkaError

from config import KAFKA_BOOTSTRAP_SERVERS, KAFKA_TOPIC, OUTPUTS_PATH, PROCESSED_DATA_PATH
from src.spark.spark_session import create_spark_session

st.set_page_config(page_title="PowerPulse | Energy Analytics", page_icon="âš¡", layout="wide")
st.markdown(
    """
    <style>
    .stApp { background: #f3f6fb; color: #11213c; }
    [data-testid="stSidebar"] { background: #0b1830; }
    [data-testid="stSidebar"] * { color: #e7efff !important; }
    [data-testid="stMetric"] { background: #ffffff; border: 1px solid #dfe7f2;
      padding: 16px 18px; border-radius: 12px; box-shadow: 0 2px 8px #1024400c; }
    h1, h2, h3 { color: #11213c; }
    </style>
    """,
    unsafe_allow_html=True,
)

PAGES = [
    "Overview",
    "Consumption Analytics",
    "Client Analytics",
    "Anomaly Detection",
    "Machine Learning",
    "Graph Analytics",
    "Streaming Analytics",
]


@st.cache_resource(show_spinner="Starting Sparkâ€¦")
def spark_session():
    return create_spark_session("PowerPulse-Streamlit-Dashboard")


@st.cache_data(show_spinner=False)
def read_json(path_text):
    path = Path(path_text)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@st.cache_data(show_spinner="Loading compact analyticsâ€¦")
def parquet_to_pandas(path_text, limit=200000):
    path = Path(path_text)
    if not path.exists():
        return pd.DataFrame()
    data = spark_session().read.parquet(str(path))
    if limit:
        data = data.limit(limit)
    return data.toPandas()


def analytics_file(name):
    return OUTPUTS_PATH / "analytics" / name


def ml_file(name):
    return OUTPUTS_PATH / "ml" / name


def section_missing(message):
    st.info(message)


def render_overview(summary):
    st.title("PowerPulse Energy Overview")
    st.caption("Distributed analysis of the UCI ElectricityLoadDiagrams20112014 dataset")
    if not summary:
        section_missing("Analytics are not available yet. Run `python scripts/run_analytics.py` first.")
        return
    overall = summary["overall_consumption"]
    peak = summary["peak_demand"]
    cols = st.columns(5)
    cols[0].metric("Total energy", f"{overall['total_energy_kwh']:,.0f} kWh")
    cols[1].metric("Average power", f"{overall['average_power_kw']:,.1f} kW")
    cols[2].metric("Peak demand", f"{peak['maximum_aggregate_demand_kw']:,.0f} kW")
    cols[3].metric("Clients", f"{overall['number_of_clients']:,}")
    cols[4].metric("Readings", f"{overall['number_of_readings']:,}")
    st.subheader("Daily system energy")
    daily = parquet_to_pandas(str(analytics_file("daily_trend")))
    if not daily.empty:
        daily["date"] = pd.to_datetime(daily["date"])
        st.plotly_chart(px.line(daily, x="date", y="total_kwh", labels={"total_kwh": "Energy (kWh)"}), width="stretch")
    else:
        section_missing("Daily trend output was not found.")
    st.caption("Source records represent historical 15-minute readings; they are not live meter data.")


def render_consumption():
    st.title("Consumption Analytics")
    daily = parquet_to_pandas(str(analytics_file("daily_trend")))
    monthly = parquet_to_pandas(str(analytics_file("monthly_trend")))
    yearly = parquet_to_pandas(str(analytics_file("yearly_trend")))
    hourly = parquet_to_pandas(str(analytics_file("hourly_trend")))
    weekend = parquet_to_pandas(str(analytics_file("weekday_weekend_trend")))
    tab1, tab2, tab3 = st.tabs(["Trends", "Hourly profile", "Calendar comparison"])
    with tab1:
        if not daily.empty:
            daily["date"] = pd.to_datetime(daily["date"])
            st.plotly_chart(px.line(daily, x="date", y="total_kwh", title="Daily energy", labels={"total_kwh": "kWh"}), width="stretch")
        if not monthly.empty:
            monthly["period"] = monthly["year"].astype(str) + "-" + monthly["month"].astype(str).str.zfill(2)
            st.plotly_chart(px.line(monthly, x="period", y="total_kwh", title="Monthly energy", labels={"total_kwh": "kWh"}), width="stretch")
        if not yearly.empty:
            st.plotly_chart(px.bar(yearly, x="year", y="total_kwh", title="Yearly energy", labels={"total_kwh": "kWh"}), width="stretch")
    with tab2:
        if not hourly.empty:
            st.plotly_chart(px.line(hourly, x="hour", y="average_kw", markers=True, title="Mean client load by hour", labels={"hour": "Hour of day", "average_kw": "Average kW"}), width="stretch")
    with tab3:
        if not weekend.empty:
            weekend["period"] = weekend["is_weekend"].map({True: "Weekend", False: "Weekday"})
            st.plotly_chart(px.bar(weekend, x="period", y="average_kw", color="period", title="Mean reading power", labels={"average_kw": "Average kW"}), width="stretch")


def render_clients():
    st.title("Client Analytics")
    clients = parquet_to_pandas(str(analytics_file("client_profiles")))
    if clients.empty:
        section_missing("Client profiles are unavailable. Run the distributed analytics pipeline.")
        return
    clients = clients.sort_values("total_kwh", ascending=False)
    selected = st.multiselect("Compare clients", clients["client_id"].tolist(), default=clients.head(5)["client_id"].tolist())
    left, right = st.columns(2)
    with left:
        st.plotly_chart(px.bar(clients.head(20), x="client_id", y="total_kwh", title="Top 20 clients by energy", labels={"total_kwh": "kWh"}), width="stretch")
    with right:
        st.plotly_chart(px.histogram(clients, x="total_kwh", nbins=30, title="Client energy distribution", labels={"total_kwh": "kWh per client"}), width="stretch")
    if selected:
        st.plotly_chart(px.bar(clients[clients.client_id.isin(selected)], x="client_id", y=["average_kw", "maximum_kw"], barmode="group", title="Selected client load comparison"), width="stretch")
    st.dataframe(clients.head(20), width="stretch", hide_index=True)


def render_anomalies(summary):
    st.title("Anomaly Detection")
    if summary:
        st.metric("Flagged readings", f"{summary.get('anomaly_reading_count', 0):,}")
        st.caption(summary.get("anomaly_interpretation", "Statistical flags for investigation."))
    anomalies = parquet_to_pandas(str(analytics_file("anomalies")), limit=50000)
    if anomalies.empty:
        section_missing("Anomaly output is unavailable.")
        return
    client_counts = anomalies.groupby("client_id").size().sort_values(ascending=False).head(20).rename("flagged_readings").reset_index()
    st.plotly_chart(px.bar(client_counts, x="client_id", y="flagged_readings", title="Clients with the most flags"), width="stretch")
    st.dataframe(anomalies.head(500), width="stretch", hide_index=True)
    st.caption("Z-score flags identify unusual client-relative readings; they do not diagnose equipment failure.")


def render_ml():
    st.title("Machine Learning")
    comparison = read_json(str(ml_file("model_comparison.json")))
    clusters = parquet_to_pandas(str(ml_file("client_clusters")))
    predictions = parquet_to_pandas(str(ml_file("linear_regression_predictions")), limit=50000)
    if comparison:
        st.subheader("Regression performance")
        st.dataframe(pd.DataFrame(comparison.get("regression_by_rmse", [])), width="stretch", hide_index=True)
        st.subheader("Classification models")
        st.dataframe(pd.DataFrame(comparison.get("classification_by_f1_high", [])), width="stretch", hide_index=True)
    if not predictions.empty:
        st.subheader("Held-out demand: actual versus prediction")
        time_col = "timestamp" if "timestamp" in predictions.columns else predictions.columns[0]
        actual_col = "label" if "label" in predictions.columns else "actual_kw"
        prediction_col = "prediction"
        if actual_col in predictions and prediction_col in predictions:
            predictions[time_col] = pd.to_datetime(predictions[time_col], errors="coerce")
            plot_data = predictions.sort_values(time_col).head(5000)
            st.plotly_chart(px.line(plot_data, x=time_col, y=[actual_col, prediction_col], title="Chronological holdout"), width="stretch")
    if not clusters.empty:
        st.subheader("Client cluster assignments")
        group_col = "prediction" if "prediction" in clusters else "cluster"
        st.plotly_chart(px.histogram(clusters, x=group_col, title="Clients per cluster"), width="stretch")
        st.dataframe(clusters, width="stretch", hide_index=True)
    if not comparison:
        section_missing("Model outputs are unavailable. Run `python scripts/train_models.py`.")


def render_graph():
    st.title("Graph Analytics")
    summary = read_json(str(OUTPUTS_PATH / "graphx" / "graph_summary.json"))
    node_files = list((OUTPUTS_PATH / "graphx" / "nodes").glob("part-*.csv"))
    edge_files = list((OUTPUTS_PATH / "graphx" / "edges").glob("part-*.csv"))
    if not summary or not node_files:
        section_missing("GraphX output is unavailable. Build and run the Scala GraphX module.")
        return
    nodes = pd.read_csv(node_files[0])
    edges = pd.read_csv(edge_files[0]) if edge_files else pd.DataFrame()
    cols = st.columns(4)
    cols[0].metric("Clients", summary["vertex_count"])
    cols[1].metric("Similarity links", f"{summary['edge_count']:,}")
    cols[2].metric("Components", summary["connected_component_count"])
    cols[3].metric("Maximum degree", summary["maximum_degree"])
    st.caption(f"Cosine threshold {summary['minimum_cosine_similarity']}; graph is behavioral similarity, not physical grid topology.")
    left, right = st.columns(2)
    with left:
        st.plotly_chart(px.histogram(nodes, x="degree", nbins=25, title="Client similarity degree distribution"), width="stretch")
    with right:
        st.plotly_chart(px.bar(nodes.nlargest(15, "page_rank"), x="client_id", y="page_rank", title="Top clients by PageRank"), width="stretch")
    if not edges.empty:
        focus = set(nodes.nlargest(30, "degree")["client_id"])
        sample = edges[edges.source_client_id.isin(focus) & edges.target_client_id.isin(focus)]
        if len(sample) > 250:
            sample = sample.nlargest(250, "cosine_similarity")
        coords = {
            client: (math.cos(2 * math.pi * i / max(len(focus), 1)), math.sin(2 * math.pi * i / max(len(focus), 1)))
            for i, client in enumerate(sorted(focus))
        }
        edge_x, edge_y = [], []
        for edge in sample.itertuples(index=False):
            sx, sy = coords[edge.source_client_id]
            tx, ty = coords[edge.target_client_id]
            edge_x.extend([sx, tx, None])
            edge_y.extend([sy, ty, None])
        network = go.Figure()
        network.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=0.5, color="#8ea4c4"), hoverinfo="skip"))
        focus_nodes = nodes[nodes.client_id.isin(focus)]
        network.add_trace(go.Scatter(
            x=[coords[c][0] for c in focus_nodes.client_id],
            y=[coords[c][1] for c in focus_nodes.client_id],
            mode="markers+text", text=focus_nodes.client_id, textposition="top center",
            marker=dict(size=10, color=focus_nodes.degree, colorscale="Blues", showscale=True,
                        colorbar=dict(title="Degree")),
            hovertemplate="%{text}<extra></extra>"))
        network.update_layout(title="Strongest links within the 30 highest-degree clients", showlegend=False,
                             xaxis=dict(visible=False), yaxis=dict(visible=False),
                             margin=dict(l=10, r=10, t=50, b=10), height=520)
        st.plotly_chart(network, width="stretch")
        st.dataframe(sample, width="stretch", hide_index=True)


def kafka_available():
    consumer = None
    try:
        consumer = KafkaConsumer(
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            request_timeout_ms=1500,
            api_version_auto_timeout_ms=1500,
            consumer_timeout_ms=100,
        )
        return True
    except Exception:
        return False
    finally:
        if consumer is not None:
            consumer.close(autocommit=False)


def render_streaming():
    st.title("Streaming Analytics")
    available = kafka_available()
    st.metric("Kafka broker", "Available" if available else "Unavailable")
    st.caption(f"Broker: {KAFKA_BOOTSTRAP_SERVERS} Â· Topic: {KAFKA_TOPIC}")
    if not available:
        st.info("Start Kafka and create the topic to enable replay. This demo replays historical dataset readings; it is not connected to live meters.")
        st.code(".\\scripts\\kafka_startup.ps1 -Action start\n.\\scripts\\kafka_startup.ps1 -Action create-topic\n.\\scripts\\run_streaming_consumer.ps1 -Once", language="powershell")
    streaming_path = OUTPUTS_PATH / "streaming" / "microbatch_aggregates"
    aggregates = parquet_to_pandas(str(streaming_path))
    if not aggregates.empty:
        st.metric("Processed reading count", f"{int(aggregates.reading_count.sum()):,}")
        aggregates = aggregates.groupby("window_start", as_index=False).agg(
            total_kwh=("total_kwh", "sum"), reading_count=("reading_count", "sum")
        )
        st.plotly_chart(px.line(aggregates.sort_values("window_start"), x="window_start", y="total_kwh", title="Historical replay: energy per event-time window"), width="stretch")
        st.dataframe(aggregates.tail(100), width="stretch", hide_index=True)
    latest = parquet_to_pandas(str(OUTPUTS_PATH / "streaming" / "latest_readings"), limit=100)
    if not latest.empty:
        st.subheader("Latest processed readings")
        st.dataframe(latest.sort_values("timestamp", ascending=False), width="stretch", hide_index=True)
    elif available:
        section_missing("Kafka is reachable; no streaming aggregates have been saved yet.")


def main():
    with st.sidebar:
        st.title("âš¡ PowerPulse")
        st.caption("Distributed energy analytics")
        page = st.radio("Navigation", PAGES, label_visibility="collapsed")
        st.divider()
        st.caption(f"Spark source: {PROCESSED_DATA_PATH}")
    summary = read_json(str(analytics_file("analytics_summary.json")))
    renderers = {
        "Overview": lambda: render_overview(summary),
        "Consumption Analytics": render_consumption,
        "Client Analytics": render_clients,
        "Anomaly Detection": lambda: render_anomalies(summary),
        "Machine Learning": render_ml,
        "Graph Analytics": render_graph,
        "Streaming Analytics": render_streaming,
    }
    try:
        renderers[page]()
    except Exception as exc:
        st.error(f"Could not load this dashboard page: {exc}")
        st.exception(exc)


if __name__ == "__main__":
    main()
