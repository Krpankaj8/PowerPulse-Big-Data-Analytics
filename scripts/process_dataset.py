"""Run the UCI-to-Parquet preprocessing pipeline."""

import argparse
import json
import logging
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DATASET_PATH, PROCESSED_DATA_PATH, OUTPUTS_PATH  # noqa: E402
from src.spark.data_loader import load_uci_wide_dataset  # noqa: E402
from src.spark.preprocessing import add_calendar_and_energy_features, unpivot_readings  # noqa: E402
from src.spark.quality_report import summarize_readings  # noqa: E402
from src.spark.spark_session import create_spark_session  # noqa: E402


LOGGER = logging.getLogger("powerpulse.preprocessing")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DATASET_PATH, help="Original UCI TXT file")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROCESSED_DATA_PATH,
        help="Destination directory for long-format Parquet data",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=OUTPUTS_PATH / "data_quality.json",
        help="Destination JSON file for distributed quality metrics",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing Parquet destination and quality report",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    report_path = args.report.expanduser().resolve()

    if not input_path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {input_path}")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. Pass --overwrite to replace it."
        )
    if report_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Quality report already exists: {report_path}. Pass --overwrite to replace it."
        )

    LOGGER.info("Reading UCI TXT data from %s", input_path)
    spark = create_spark_session("PowerPulse-UCI-Preprocessing")
    try:
        wide = load_uci_wide_dataset(spark, input_path)
        if len(wide.columns) < 2:
            raise ValueError("The source must contain a timestamp and client columns.")
        source_timestamp_column = wide.columns[0]
        client_count = len(wide.columns) - 1
        LOGGER.info("Detected %d client columns; timestamp column is %r", client_count, source_timestamp_column)

        readings = add_calendar_and_energy_features(unpivot_readings(wide))
        quality = summarize_readings(readings)
        quality.update(
            {
                "source_file": str(input_path),
                "source_size_bytes": input_path.stat().st_size,
                "source_timestamp_column": source_timestamp_column,
                "source_row_count": quality["reading_count"] // client_count,
                "source_client_column_count": client_count,
                "processed_parquet_path": str(output_path),
                "energy_conversion": "kWh = kW * 0.25 (15-minute interval)",
                "day_of_week_convention": "Sunday=1 through Saturday=7; weekend is Sunday or Saturday",
            }
        )

        LOGGER.info(
            "Readings=%d clients=%d null_kw=%d null_timestamp=%d zero_kw=%d",
            quality["reading_count"],
            quality["client_count"],
            quality["null_kw_count"],
            quality["null_timestamp_count"],
            quality["zero_kw_count"],
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Partitioning by year forces Spark to sort all expanded rows by the
        # partition key. On a Windows laptop this can exhaust the default JVM
        # heap, so year remains a queryable column and files are bounded by row
        # count instead of using physical year directories.
        writer = readings.write.mode("overwrite" if args.overwrite else "errorifexists")
        writer.option("maxRecordsPerFile", 1_000_000).parquet(str(output_path))

        report_path.parent.mkdir(parents=True, exist_ok=True)
        if report_path.exists() and args.overwrite:
            report_path.unlink()
        report_path.write_text(json.dumps(quality, indent=2), encoding="utf-8")
        LOGGER.info("Parquet data written to %s", output_path)
        LOGGER.info("Quality report written to %s", report_path)
        print(json.dumps({key: value for key, value in quality.items() if key != "per_client"}, indent=2))
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
