"""Phase 1 smoke check: start Spark and execute a tiny distributed action."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.spark.spark_session import create_spark_session  # noqa: E402


def main() -> None:
    spark = create_spark_session("PowerPulse-Environment-Check")
    try:
        count = spark.range(10).count()
        print(f"Spark version: {spark.version}")
        print(f"Spark master: {spark.sparkContext.master}")
        print(f"Smoke-check rows: {count}")
        if count != 10:
            raise RuntimeError(f"Unexpected Spark result: expected 10 rows, got {count}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
