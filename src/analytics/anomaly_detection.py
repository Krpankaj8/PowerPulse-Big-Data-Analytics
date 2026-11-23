"""Statistical per-client anomaly screening with a three-sigma z-score."""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def detect_client_zscore_anomalies(readings: DataFrame, threshold: float = 3.0) -> DataFrame:
    """Flag unusual readings relative to each client's own load distribution.

    The client summary has only one row per client and is broadcast for the join.
    Flags identify readings for investigation; they do not assert equipment failure.
    """
    if threshold <= 0:
        raise ValueError("threshold must be positive")

    client_stats = readings.groupBy("client_id").agg(
        F.avg("kw").alias("client_mean_kw"),
        F.stddev_pop("kw").alias("client_stddev_kw"),
    )
    scored = (
        readings.filter(F.col("kw").isNotNull())
        .join(F.broadcast(client_stats), on="client_id", how="inner")
        .withColumn(
            "z_score",
            F.when(
                F.col("client_stddev_kw") > 0,
                (F.col("kw") - F.col("client_mean_kw")) / F.col("client_stddev_kw"),
            ),
        )
    )
    return scored.filter(F.abs("z_score") > F.lit(threshold))
