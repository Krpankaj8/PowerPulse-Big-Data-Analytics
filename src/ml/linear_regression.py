"""Spark MLlib linear regression for chronological demand forecasting."""

from pyspark.ml import Pipeline
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.ml.regression import LinearRegression

from src.ml.features import DEMAND_FEATURE_COLUMNS


def create_linear_regression_pipeline() -> Pipeline:
    assembler = VectorAssembler(
        inputCols=DEMAND_FEATURE_COLUMNS,
        outputCol="raw_features",
        handleInvalid="error",
    )
    scaler = StandardScaler(
        inputCol="raw_features", outputCol="features", withMean=True, withStd=True
    )
    estimator = LinearRegression(
        featuresCol="features",
        labelCol="label",
        predictionCol="prediction",
        maxIter=100,
        regParam=0.0,
        elasticNetParam=0.0,
    )
    return Pipeline(stages=[assembler, scaler, estimator])
