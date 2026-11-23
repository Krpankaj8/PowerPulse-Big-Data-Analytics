"""Distributed data-quality summary functions for the processed readings."""

from typing import Any

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def summarize_readings(readings: DataFrame) -> dict[str, Any]:
    """Count rows, clients, nulls, zeros and observed timestamp bounds.

    Only one aggregate row and one compact per-client summary are brought to the
    driver; the full readings table remains distributed in Spark.
    """
    summary_row = readings.agg(
        F.count(F.lit(1)).alias("reading_count"),
        F.countDistinct("client_id").alias("client_count"),
        F.sum(F.when(F.col("kw").isNull(), 1).otherwise(0)).alias("null_kw_count"),
        F.sum(F.when(F.col("timestamp").isNull(), 1).otherwise(0)).alias(
            "null_timestamp_count"
        ),
        F.sum(F.when(F.col("kw") == 0.0, 1).otherwise(0)).alias("zero_kw_count"),
        F.date_format(F.min("timestamp"), "yyyy-MM-dd'T'HH:mm:ss").alias(
            "min_timestamp"
        ),
        F.date_format(F.max("timestamp"), "yyyy-MM-dd'T'HH:mm:ss").alias(
            "max_timestamp"
        ),
    ).first()

    per_client_rows = (
        readings.groupBy("client_id")
        .agg(
            F.sum(F.when(F.col("kw").isNull(), 1).otherwise(0)).alias("null_kw_count"),
            F.sum(F.when(F.col("kw") == 0.0, 1).otherwise(0)).alias("zero_kw_count"),
        )
        .orderBy("client_id")
        .collect()
    )
    per_client = {
        row["client_id"]: {
            "null_kw_count": int(row["null_kw_count"] or 0),
            "zero_kw_count": int(row["zero_kw_count"] or 0),
        }
        for row in per_client_rows
    }
    return {
        "reading_count": int(summary_row["reading_count"] or 0),
        "client_count": int(summary_row["client_count"] or 0),
        "null_kw_count": int(summary_row["null_kw_count"] or 0),
        "null_timestamp_count": int(summary_row["null_timestamp_count"] or 0),
        "zero_kw_count": int(summary_row["zero_kw_count"] or 0),
        "min_timestamp": summary_row["min_timestamp"],
        "max_timestamp": summary_row["max_timestamp"],
        "per_client": per_client,
    }
