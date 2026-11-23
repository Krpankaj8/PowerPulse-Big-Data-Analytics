"""Overall electricity consumption metrics computed with Spark SQL."""

from typing import Any

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def summarize_consumption(readings: DataFrame) -> dict[str, Any]:
    """Return compact project-wide metrics without collecting source rows."""
    row = readings.agg(
        F.sum("kwh").alias("total_kwh"),
        F.avg("kw").alias("average_kw"),
        F.max("kw").alias("maximum_kw"),
        F.min("kw").alias("minimum_kw"),
        F.countDistinct("client_id").alias("client_count"),
        F.count(F.lit(1)).alias("reading_count"),
    ).first()
    return {
        "total_energy_kwh": float(row["total_kwh"] or 0.0),
        "average_power_kw": float(row["average_kw"] or 0.0),
        "maximum_client_power_kw": float(row["maximum_kw"] or 0.0),
        "minimum_client_power_kw": float(row["minimum_kw"] or 0.0),
        "number_of_clients": int(row["client_count"] or 0),
        "number_of_readings": int(row["reading_count"] or 0),
    }
