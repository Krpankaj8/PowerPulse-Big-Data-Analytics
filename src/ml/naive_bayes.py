"""Spark MLlib multinomial Naive Bayes for nonnegative temporal/load features."""

from pyspark.ml import Pipeline
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.feature import VectorAssembler

from src.ml.features import DEMAND_FEATURE_COLUMNS


def create_naive_bayes_pipeline() -> Pipeline:
    assembler = VectorAssembler(inputCols=DEMAND_FEATURE_COLUMNS, outputCol="features")
    estimator = NaiveBayes(
        featuresCol="features",
        labelCol="label",
        weightCol="class_weight",
        predictionCol="prediction",
        modelType="multinomial",
        smoothing=1.0,
    )
    return Pipeline(stages=[assembler, estimator])
