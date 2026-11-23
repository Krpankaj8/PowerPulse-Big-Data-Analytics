"""Client-level consumption profiles and compact top/bottom comparisons."""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def client_profiles(readings: DataFrame) -> DataFrame:
    """Aggregate load level and variability for each client."""
    return readings.groupBy("client_id").agg(
        F.sum("kwh").alias("total_kwh"),
        F.avg("kw").alias("average_kw"),
        F.max("kw").alias("maximum_kw"),
        F.min("kw").alias("minimum_kw"),
        F.stddev_pop("kw").alias("load_stddev_kw"),
        F.count(F.lit(1)).alias("reading_count"),
    )


def top_clients(profiles: DataFrame, limit: int = 10, descending: bool = True) -> list[dict]:
    """Collect only a small ranked client list for reports and dashboard cards."""
    if limit < 1:
        raise ValueError("limit must be at least 1")
    direction = F.desc("total_kwh") if descending else F.asc("total_kwh")
    return [row.asDict() for row in profiles.orderBy(direction, "client_id").limit(limit).collect()]
