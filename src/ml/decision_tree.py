"""Weighted Spark MLlib decision-tree high-demand classifier."""

from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import VectorAssembler

from src.ml.features import DEMAND_FEATURE_COLUMNS


def create_decision_tree_pipeline() -> Pipeline:
    assembler = VectorAssembler(inputCols=DEMAND_FEATURE_COLUMNS, outputCol="features")
    estimator = DecisionTreeClassifier(
        featuresCol="features",
        labelCol="label",
        weightCol="class_weight",
        predictionCol="prediction",
        maxDepth=8,
        seed=42,
    )
    return Pipeline(stages=[assembler, estimator])
