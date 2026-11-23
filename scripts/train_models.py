"""Train and evaluate Spark MLlib forecasting, classification and clustering models."""

import argparse
from functools import reduce
import json
import logging
from pathlib import Path
import sys

from pyspark.ml import PipelineModel
from pyspark.sql import functions as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import MODELS_PATH, OUTPUTS_PATH, PROCESSED_DATA_PATH  # noqa: E402
from src.ml.clustering import build_client_feature_table, fit_kmeans_profiles  # noqa: E402
from src.ml.decision_tree import create_decision_tree_pipeline  # noqa: E402
from src.ml.evaluation import binary_classification_metrics, regression_metrics  # noqa: E402
from src.ml.features import (  # noqa: E402
    DEMAND_FEATURE_COLUMNS,
    add_balanced_class_weights,
    build_demand_features,
    chronological_train_test_split,
)
from src.ml.linear_regression import create_linear_regression_pipeline  # noqa: E402
from src.ml.logistic_regression import create_logistic_regression_pipeline  # noqa: E402
from src.ml.model_comparison import compare_models  # noqa: E402
from src.ml.naive_bayes import create_naive_bayes_pipeline  # noqa: E402
from src.spark.spark_session import create_spark_session  # noqa: E402


LOGGER = logging.getLogger("powerpulse.ml")
CLASSIFIERS = {
    "logistic_regression": create_logistic_regression_pipeline,
    "decision_tree": create_decision_tree_pipeline,
    "naive_bayes": create_naive_bayes_pipeline,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=PROCESSED_DATA_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUTS_PATH / "ml")
    parser.add_argument("--models", type=Path, default=MODELS_PATH / "ml")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing ML outputs and models")
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--high-demand-quantile", type=float, default=0.9)
    return parser.parse_args()


def _save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _save_frame(frame, path: Path, overwrite: bool) -> None:
    frame.write.mode("overwrite" if overwrite else "errorifexists").option(
        "maxRecordsPerFile", 1_000_000
    ).parquet(str(path))


def _save_model(model, path: Path, overwrite: bool) -> None:
    writer = model.write()
    if overwrite:
        writer = writer.overwrite()
    writer.save(str(path))


def _class_counts(data) -> dict[str, int]:
    counts = {float(row["label"]): int(row["count"]) for row in data.groupBy("label").count().collect()}
    return {"normal": counts.get(0.0, 0), "high": counts.get(1.0, 0)}


def train_regression(split, output_path: Path, model_root: Path, args) -> dict:
    cutoff_text = split.cutoff_timestamp.strftime("%Y-%m-%d %H:%M:%S")

    LOGGER.info("Fitting scaled Spark LinearRegression on %d chronological rows", split.train_rows)
    pipeline_model = create_linear_regression_pipeline().fit(split.train)
    predictions = pipeline_model.transform(split.test).select(
        "timestamp", "label", "prediction"
    )
    metrics = regression_metrics(predictions)
    metrics.update(
        {
            "train_rows": split.train_rows,
            "test_rows": split.test_rows,
            "train_fraction": args.train_fraction,
            "cutoff_timestamp": cutoff_text,
            "target": "aggregate system demand in kW",
        }
    )
    _save_model(pipeline_model, model_root / "linear_regression", args.overwrite)
    _save_frame(predictions, output_path / "linear_regression_predictions", args.overwrite)
    _save_json(output_path / "regression_metrics.json", {"linear_regression": metrics})
    return {"linear_regression": metrics}


def train_classifiers(split, output_path: Path, model_root: Path, args) -> dict:
    if not 0.5 < args.high_demand_quantile < 1.0:
        raise ValueError("high_demand_quantile must be between 0.5 and 1.0")
    threshold_row = split.train.agg(
        F.percentile_approx("label", args.high_demand_quantile, 10_000).alias("threshold")
    ).first()
    high_threshold_kw = float(threshold_row["threshold"])
    train_labeled = split.train.withColumn(
        "label", (F.col("label") >= F.lit(high_threshold_kw)).cast("double")
    )
    test_labeled = split.test.withColumn(
        "label", (F.col("label") >= F.lit(high_threshold_kw)).cast("double")
    )
    weighted_train, class_weights = add_balanced_class_weights(train_labeled)
    train_class_counts = _class_counts(train_labeled)
    test_class_counts = _class_counts(test_labeled)
    cutoff_text = split.cutoff_timestamp.strftime("%Y-%m-%d %H:%M:%S")

    classifier_results = {}
    prediction_frames = []
    for name, pipeline_factory in CLASSIFIERS.items():
        LOGGER.info("Fitting %s", name)
        model = pipeline_factory().fit(weighted_train)
        predictions = model.transform(test_labeled).select(
            "timestamp", "label", "prediction"
        )
        metrics = binary_classification_metrics(predictions)
        metrics.update(
            {
                "high_demand_threshold_kw": high_threshold_kw,
                "threshold_quantile_train_only": args.high_demand_quantile,
                "train_rows": split.train_rows,
                "test_rows": split.test_rows,
                "chronological_cutoff": cutoff_text,
                "train_class_counts": train_class_counts,
                "test_class_counts": test_class_counts,
                "balanced_training_weights": class_weights,
            }
        )
        if name == "decision_tree":
            tree_model = model.stages[-1]
            importance_values = tree_model.featureImportances.toArray().tolist()
            metrics["feature_importance"] = [
                {"feature": feature, "importance": float(importance)}
                for feature, importance in sorted(
                    zip(DEMAND_FEATURE_COLUMNS, importance_values),
                    key=lambda pair: pair[1],
                    reverse=True,
                )
            ]
        classifier_results[name] = metrics
        _save_model(model, model_root / name, args.overwrite)
        prediction_frames.append(predictions.withColumn("model", F.lit(name)))

    combined_predictions = reduce(lambda left, right: left.unionByName(right), prediction_frames)
    _save_frame(combined_predictions, output_path / "classification_predictions", args.overwrite)
    _save_json(output_path / "classification_metrics.json", classifier_results)
    return classifier_results


def train_client_clusters(readings, output_path: Path, model_root: Path, args) -> dict:
    LOGGER.info("Building and scaling client-level hourly load profiles")
    client_features, feature_columns = build_client_feature_table(readings)
    model, scored, details = fit_kmeans_profiles(client_features, feature_columns)
    assignments = scored.select(
        "client_id",
        F.col("prediction").alias("cluster"),
        "average_kw",
        "maximum_kw",
        "minimum_kw",
        "load_stddev_kw",
    )
    characteristics = assignments.groupBy("cluster").agg(
        F.count(F.lit(1)).alias("client_count"),
        F.avg("average_kw").alias("cluster_average_kw"),
        F.avg("maximum_kw").alias("cluster_maximum_kw"),
        F.avg("load_stddev_kw").alias("cluster_average_stddev_kw"),
    ).orderBy("cluster")
    distribution = [row.asDict() for row in characteristics.collect()]
    details["cluster_distribution_and_characteristics"] = distribution
    details["feature_columns"] = feature_columns
    _save_model(model, model_root / "kmeans_clients", args.overwrite)
    _save_frame(assignments, output_path / "client_clusters", args.overwrite)
    _save_frame(characteristics, output_path / "cluster_characteristics", args.overwrite)
    _save_json(output_path / "clustering_metrics.json", details)
    return details


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    model_root = args.models.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Processed Parquet dataset not found: {input_path}")
    if not 0.5 < args.train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0.5 and 1.0")
    for destination in (output_path, model_root):
        if destination.exists() and not args.overwrite:
            raise FileExistsError(
                f"Destination already exists: {destination}. Pass --overwrite to replace it."
            )
    output_path.mkdir(parents=True, exist_ok=True)
    model_root.mkdir(parents=True, exist_ok=True)

    spark = create_spark_session("PowerPulse-Spark-MLlib")
    try:
        readings = spark.read.parquet(str(input_path))
        LOGGER.info("Building shared time-ordered demand features")
        demand_features = build_demand_features(readings).cache()
        split = chronological_train_test_split(demand_features, args.train_fraction)
        clustering = train_client_clusters(readings, output_path, model_root, args)
        regression = train_regression(split, output_path, model_root, args)
        classification = train_classifiers(split, output_path, model_root, args)
        comparison = compare_models(regression, classification)
        comparison["target_definition"] = {
            "classification": "High demand if aggregate system kW is at or above the training-only quantile threshold.",
            "high_demand_quantile": args.high_demand_quantile,
        }
        comparison["clustering"] = clustering
        _save_json(output_path / "model_comparison.json", comparison)
        print(json.dumps(comparison, indent=2))
        LOGGER.info("Models, predictions and evaluations saved to %s and %s", output_path, model_root)
    finally:
        if "demand_features" in locals():
            demand_features.unpersist()
        spark.stop()


if __name__ == "__main__":
    main()
