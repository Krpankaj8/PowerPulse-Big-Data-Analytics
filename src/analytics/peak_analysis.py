"""Aggregate demand and peak load analysis."""

from typing import Any

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def aggregate_demand_by_timestamp(readings: DataFrame) -> DataFrame:
    """Sum client kW values at each timestamp to get system demand in kW."""
    return readings.groupBy("timestamp").agg(
        F.sum("kw").alias("aggregate_demand_kw")
    )


def peak_summary(readings: DataFrame, top_n: int = 20) -> tuple[dict[str, Any], DataFrame]:
    """Return compact peak metrics and the highest-demand intervals."""
    if top_n < 1:
        raise ValueError("top_n must be at least 1")
    demand = aggregate_demand_by_timestamp(readings)
    bounds = demand.agg(
        F.max("aggregate_demand_kw").alias("maximum_aggregate_demand_kw"),
        F.avg("aggregate_demand_kw").alias("average_aggregate_demand_kw"),
    ).first()
    top_timestamps = [
        {
            "timestamp": row["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
            "aggregate_demand_kw": float(row["aggregate_demand_kw"]),
        }
        for row in demand.orderBy(F.desc("aggregate_demand_kw"), F.asc("timestamp"))
        .limit(top_n)
        .collect()
    ]
    by_hour = demand.withColumn("hour", F.hour("timestamp")).groupBy("hour").agg(
        F.avg("aggregate_demand_kw").alias("average_aggregate_demand_kw"),
        F.max("aggregate_demand_kw").alias("maximum_aggregate_demand_kw"),
    ).orderBy("hour")
    return (
        {
            "maximum_aggregate_demand_kw": float(
                bounds["maximum_aggregate_demand_kw"] or 0.0
            ),
            "average_aggregate_demand_kw": float(
                bounds["average_aggregate_demand_kw"] or 0.0
            ),
            "top_demand_timestamps": top_timestamps,
        },
        by_hour,
    )
