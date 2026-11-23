"""Create the shared SparkSession used by PowerPulse modules."""

import sys
import os

from pyspark.sql import SparkSession

from config import SPARK_APP_NAME, SPARK_MASTER, SPARK_TIMEZONE


def create_spark_session(app_name: str | None = None) -> SparkSession:
    """Create or reuse a configured Spark session.

    The local master makes the project runnable on a Windows development
    machine. Set POWERPULSE_SPARK_MASTER to a cluster URL when one is available.
    """
    # Spark's Python worker launcher on Windows reads this environment variable.
    # Honor an explicit user override, otherwise use the currently active venv.
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

    return (
        SparkSession.builder.appName(app_name or SPARK_APP_NAME)
        .master(SPARK_MASTER)
        .config("spark.sql.session.timeZone", SPARK_TIMEZONE)
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.ui.enabled", "false")
        .config("spark.pyspark.python", sys.executable)
        .config("spark.pyspark.driver.python", sys.executable)
        .getOrCreate()
    )
