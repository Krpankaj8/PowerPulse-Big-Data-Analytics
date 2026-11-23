"""Run Spark SQL, RDD and DataFrame analytics over processed UCI readings."""

import argparse
import json
import logging
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import OUTPUTS_PATH, PROCESSED_DATA_PATH  # noqa: E402
from src.analytics.anomaly_detection import detect_client_zscore_anomalies  # noqa: E402
from src.analytics.client_analysis import client_profiles, top_clients  # noqa: E402
from src.analytics.consumption_analysis import summarize_consumption  # noqa: E402
from src.analytics.distributed_demo import rdd_top_clients_by_energy, spark_sql_summary  # noqa: E402
from src.analytics.peak_analysis import peak_summary  # noqa: E402
from src.analytics.trend_analysis import (  # noqa: E402
    daily_trend,
    hourly_trend,
    monthly_trend,
    weekday_weekend_trend,
    yearly_trend,
)
from src.spark.spark_session import create_spark_session  # noqa: E402


LOGGER = logging.getLogger("powerpulse.analytics")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=PROCESSED_DATA_PATH, help="Processed Parquet path")
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUTS_PATH / "analytics",
        help="Directory for analytical Parquet and JSON outputs",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output directory")
    parser.add_argument("--anomaly-z-threshold", type=float, default=3.0)
    return parser.parse_args()


def write_parquet(frame, path: Path, overwrite: bool) -> None:
    mode = "overwrite" if overwrite else "errorifexists"
    frame.write.mode(mode).option("maxRecordsPerFile", 1_000_000).parquet(str(path))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Processed Parquet dataset not found: {input_path}")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Analytics output already exists: {output_path}. Pass --overwrite to replace it."
        )
    output_path.mkdir(parents=True, exist_ok=True)

    spark = create_spark_session("PowerPulse-Distributed-Analytics")
    try:
        LOGGER.info("Loading processed readings from %s", input_path)
        readings = spark.read.parquet(str(input_path))

        LOGGER.info("Computing project-wide consumption summary")
        overall = summarize_consumption(readings)
        trend_frames = {
            "daily_trend": daily_trend(readings),
            "monthly_trend": monthly_trend(readings),
            "yearly_trend": yearly_trend(readings),
            "hourly_trend": hourly_trend(readings),
            "weekday_weekend_trend": weekday_weekend_trend(readings),
        }
        for name, frame in trend_frames.items():
            LOGGER.info("Writing %s", name)
            write_parquet(frame, output_path / name, args.overwrite)

        LOGGER.info("Computing peak demand")
        peaks, average_demand_by_hour = peak_summary(readings)
        write_parquet(average_demand_by_hour, output_path / "aggregate_demand_by_hour", args.overwrite)

        LOGGER.info("Computing client profiles and per-client z-score flags")
        profiles = client_profiles(readings)
        top = top_clients(profiles, limit=10, descending=True)
        bottom = top_clients(profiles, limit=10, descending=False)
        anomalies = detect_client_zscore_anomalies(readings, args.anomaly_z_threshold)
        anomaly_counts = anomalies.groupBy("client_id").count().withColumnRenamed(
            "count", "anomaly_count"
        )
        client_results = profiles.join(anomaly_counts, on="client_id", how="left").fillna(
            {"anomaly_count": 0}
        )
        write_parquet(client_results, output_path / "client_profiles", args.overwrite)
        write_parquet(anomalies, output_path / "anomalies", args.overwrite)
        anomaly_count = anomalies.count()

        LOGGER.info("Running RDD and Spark SQL demonstrations")
        rdd_top = rdd_top_clients_by_energy(readings, limit=10)
        sql_results = spark_sql_summary(spark, readings)

        report = {
            "overall_consumption": overall,
            "peak_demand": peaks,
            "top_clients_by_energy": top,
            "lowest_clients_by_energy": bottom,
            "rdd_top_clients_by_energy": rdd_top,
            "spark_sql": sql_results,
            "anomaly_method": "Per-client z-score; absolute z-score greater than the configured threshold",
            "anomaly_z_threshold": args.anomaly_z_threshold,
            "anomaly_reading_count": int(anomaly_count),
            "anomaly_interpretation": "Statistical screening flags for investigation, not equipment-failure labels.",
        }
        report_file = output_path / "analytics_summary.json"
        if report_file.exists() and not args.overwrite:
            raise FileExistsError(f"Analytics summary already exists: {report_file}")
        report_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
        LOGGER.info("Analytics outputs saved under %s", output_path)
        print(json.dumps({key: value for key, value in report.items() if key != "spark_sql"}, indent=2))
        print(f"Spark SQL year aggregates: {json.dumps(sql_results['yearly'])}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
