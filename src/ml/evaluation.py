"""Spark-native regression and binary classification metrics."""

from typing import Any

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def regression_metrics(predictions: DataFrame) -> dict[str, float]:
    """Calculate MAE, RMSE and R-squared with MLlib evaluators."""
    result = {}
    for metric in ("mae", "rmse", "r2"):
        evaluator = RegressionEvaluator(
            labelCol="label", predictionCol="prediction", metricName=metric
        )
        result[metric.upper()] = float(evaluator.evaluate(predictions))
    return result


def binary_classification_metrics(predictions: DataFrame) -> dict[str, Any]:
    """Compute high-demand positive-class metrics and a 2x2 confusion matrix."""
    row = predictions.agg(
        F.sum(F.when((F.col("label") == 1.0) & (F.col("prediction") == 1.0), 1).otherwise(0)).alias(
            "tp"
        ),
        F.sum(F.when((F.col("label") == 0.0) & (F.col("prediction") == 0.0), 1).otherwise(0)).alias(
            "tn"
        ),
        F.sum(F.when((F.col("label") == 0.0) & (F.col("prediction") == 1.0), 1).otherwise(0)).alias(
            "fp"
        ),
        F.sum(F.when((F.col("label") == 1.0) & (F.col("prediction") == 0.0), 1).otherwise(0)).alias(
            "fn"
        ),
    ).first()
    tp, tn, fp, fn = (int(row[name] or 0) for name in ("tp", "tn", "fp", "fn"))
    total = tp + tn + fp + fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "accuracy": (tp + tn) / total if total else 0.0,
        "precision_high": precision,
        "recall_high": recall,
        "f1_high": 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "confusion_matrix": {"labels": ["normal", "high"], "matrix": [[tn, fp], [fn, tp]]},
        "test_count": total,
    }
