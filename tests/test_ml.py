"""Test-only rows for chronological ML splits, class balance and model metrics."""

from datetime import datetime, timedelta
import unittest

from pyspark.sql.types import DoubleType, StructField, StructType, TimestampNTZType

from src.ml.evaluation import binary_classification_metrics, regression_metrics
from src.ml.features import add_balanced_class_weights, chronological_train_test_split
from src.spark.spark_session import create_spark_session


class MLUtilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.spark = create_spark_session("PowerPulse-ML-Unit-Tests")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.spark.stop()

    def test_chronological_split_keeps_future_rows_in_test(self) -> None:
        start = datetime(2011, 1, 1)
        rows = [(start + timedelta(minutes=15 * index), float(index)) for index in range(10)]
        schema = StructType(
            [
                StructField("timestamp", TimestampNTZType(), False),
                StructField("label", DoubleType(), False),
            ]
        )
        split = chronological_train_test_split(
            self.spark.createDataFrame(rows, schema), train_fraction=0.8
        )
        self.assertEqual(split.train_rows, 8)
        self.assertEqual(split.test_rows, 2)
        self.assertLessEqual(split.train.agg({"timestamp": "max"}).first()[0], split.cutoff_timestamp)
        self.assertGreater(split.test.agg({"timestamp": "min"}).first()[0], split.cutoff_timestamp)

    def test_balanced_weights_and_binary_metrics(self) -> None:
        class_rows = [(0.0,)] * 10 + [(1.0,)] * 2
        classes = self.spark.createDataFrame(class_rows, ["label"])
        weighted, details = add_balanced_class_weights(classes)
        self.assertAlmostEqual(details["normal_weight"], 0.6)
        self.assertAlmostEqual(details["high_weight"], 3.0)
        sums = weighted.groupBy("label").sum("class_weight").collect()
        self.assertAlmostEqual(sums[0]["sum(class_weight)"], sums[1]["sum(class_weight)"])

        predictions = self.spark.createDataFrame(
            [(1.0, 1.0), (1.0, 0.0), (0.0, 1.0), (0.0, 0.0)],
            ["label", "prediction"],
        )
        metrics = binary_classification_metrics(predictions)
        self.assertAlmostEqual(metrics["accuracy"], 0.5)
        self.assertAlmostEqual(metrics["precision_high"], 0.5)
        self.assertAlmostEqual(metrics["recall_high"], 0.5)
        self.assertEqual(metrics["confusion_matrix"]["matrix"], [[1, 1], [1, 1]])

    def test_regression_metric_keys(self) -> None:
        predictions = self.spark.createDataFrame(
            [(1.0, 1.0), (2.0, 1.0), (3.0, 3.0)],
            ["label", "prediction"],
        )
        metrics = regression_metrics(predictions)
        self.assertEqual(set(metrics), {"MAE", "RMSE", "R2"})
        self.assertAlmostEqual(metrics["MAE"], 1.0 / 3.0)


if __name__ == "__main__":
    unittest.main()
