"""Leakage-aware demand feature creation and chronological splitting."""

from dataclasses import dataclass
import math
from typing import Any

from pyspark.sql import DataFrame
from pyspark.sql import Window
from pyspark.sql import functions as F


DEMAND_FEATURE_COLUMNS = [
    "hour",
    "day_of_week",
    "month",
    "weekend_flag",
    "lag_1_kw",
    "lag_96_kw",
]


@dataclass(frozen=True)
class ChronologicalSplit:
    train: DataFrame
    test: DataFrame
    cutoff_timestamp: Any
    total_rows: int
    train_rows: int
    test_rows: int


def build_demand_features(readings: DataFrame) -> DataFrame:
    """Aggregate system load and add calendar and past-only load features.

    ``lag_1_kw`` uses the previous 15-minute interval and ``lag_96_kw`` uses
    the previous 96th observation (a nominal day under 15-minute sampling).
    Both are strictly prior observations; no target-period data is used.
    """
    demand = readings.groupBy("timestamp").agg(
        F.sum("kw").alias("aggregate_demand_kw"),
        F.first("hour", ignorenulls=True).alias("hour"),
        F.first("day_of_week", ignorenulls=True).alias("day_of_week"),
        F.first("month", ignorenulls=True).alias("month"),
        F.first("is_weekend", ignorenulls=True).alias("is_weekend"),
    )
    chronological_window = Window.orderBy("timestamp")
    return (
        demand.withColumn("lag_1_kw", F.lag("aggregate_demand_kw", 1).over(chronological_window))
        .withColumn("lag_96_kw", F.lag("aggregate_demand_kw", 96).over(chronological_window))
        .withColumn("weekend_flag", F.col("is_weekend").cast("double"))
        .filter(
            F.col("aggregate_demand_kw").isNotNull()
            & F.col("lag_1_kw").isNotNull()
            & F.col("lag_96_kw").isNotNull()
        )
        .withColumnRenamed("aggregate_demand_kw", "label")
        .select("timestamp", "label", *DEMAND_FEATURE_COLUMNS)
    )


def chronological_train_test_split(
    data: DataFrame, train_fraction: float = 0.8
) -> ChronologicalSplit:
    """Split timestamped rows in time order, returning only the small cutoff to Python."""
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")
    total = data.count()
    train_rows = math.floor(total * train_fraction)
    if train_rows < 1 or train_rows >= total:
        raise ValueError(f"Not enough rows for a chronological split: {total}")
    cutoff = (
        data.orderBy("timestamp").limit(train_rows).agg(F.max("timestamp").alias("cutoff")).first()[
            "cutoff"
        ]
    )
    cutoff_literal = F.to_timestamp_ntz(
        F.lit(cutoff.strftime("%Y-%m-%d %H:%M:%S"))
    )
    train = data.filter(F.col("timestamp") <= cutoff_literal)
    test = data.filter(F.col("timestamp") > cutoff_literal)
    return ChronologicalSplit(
        train=train,
        test=test,
        cutoff_timestamp=cutoff,
        total_rows=total,
        train_rows=train_rows,
        test_rows=total - train_rows,
    )


def add_balanced_class_weights(training: DataFrame) -> tuple[DataFrame, dict[str, float]]:
    """Add inverse-frequency weights calculated only from labeled training rows."""
    counts = {
        float(row["label"]): int(row["count"])
        for row in training.groupBy("label").count().collect()
    }
    negative = counts.get(0.0, 0)
    positive = counts.get(1.0, 0)
    total = negative + positive
    if not positive or not negative:
        raise ValueError("Training classification data must contain both demand classes.")
    negative_weight = total / (2.0 * negative)
    positive_weight = total / (2.0 * positive)
    weighted = training.withColumn(
        "class_weight",
        F.when(F.col("label") == 1.0, F.lit(positive_weight)).otherwise(
            F.lit(negative_weight)
        ),
    )
    return weighted, {
        "normal_train_count": negative,
        "high_train_count": positive,
        "normal_weight": negative_weight,
        "high_weight": positive_weight,
    }
