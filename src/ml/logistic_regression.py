"""Weighted Spark MLlib logistic regression pipeline."""

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import StandardScaler, VectorAssembler

from src.ml.features import DEMAND_FEATURE_COLUMNS


def create_logistic_regression_pipeline() -> Pipeline:
    assembler = VectorAssembler(inputCols=DEMAND_FEATURE_COLUMNS, outputCol="raw_features")
    scaler = StandardScaler(
        inputCol="raw_features", outputCol="features", withMean=True, withStd=True
    )
    estimator = LogisticRegression(
        featuresCol="features",
        labelCol="label",
        weightCol="class_weight",
        maxIter=100,
        regParam=0.0,
        elasticNetParam=0.0,
    )
    return Pipeline(stages=[assembler, scaler, estimator])
