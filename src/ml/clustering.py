"""Client-behaviour clustering with scaled profile features and silhouette selection."""

from pyspark.ml import PipelineModel
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def build_client_feature_table(readings: DataFrame) -> tuple[DataFrame, list[str]]:
    """Combine per-client load statistics with a 24-hour mean power profile."""
    base = readings.groupBy("client_id").agg(
        F.avg("kw").alias("average_kw"),
        F.max("kw").alias("maximum_kw"),
        F.min("kw").alias("minimum_kw"),
        F.stddev_pop("kw").alias("load_stddev_kw"),
    )
    hourly = readings.groupBy("client_id", "hour").agg(F.avg("kw").alias("hour_average_kw"))
    pivoted = hourly.groupBy("client_id").pivot("hour", list(range(24))).agg(
        F.first("hour_average_kw")
    )
    hour_columns = [str(hour) for hour in range(24)]
    pivoted = pivoted.fillna(0.0, subset=hour_columns).select(
        "client_id", *[F.col(hour).alias(f"hour_{int(hour):02d}_average_kw") for hour in hour_columns]
    )
    table = base.join(pivoted, on="client_id", how="inner")
    feature_columns = ["average_kw", "maximum_kw", "minimum_kw", "load_stddev_kw"] + [
        f"hour_{hour:02d}_average_kw" for hour in range(24)
    ]
    return table, feature_columns


def fit_kmeans_profiles(
    client_features: DataFrame,
    feature_columns: list[str],
    candidate_k: tuple[int, ...] = (2, 3, 4, 5, 6),
) -> tuple[PipelineModel, DataFrame, dict]:
    """Fit a scaler on client profiles and select k by silhouette score."""
    if not candidate_k or any(k < 2 for k in candidate_k):
        raise ValueError("candidate_k must contain values of at least 2")
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="raw_features")
    assembled = assembler.transform(client_features)
    scaler = StandardScaler(
        inputCol="raw_features", outputCol="scaled_features", withMean=True, withStd=True
    )
    scaler_model = scaler.fit(assembled)
    scaled = scaler_model.transform(assembled).cache()
    evaluator = ClusteringEvaluator(
        featuresCol="scaled_features", predictionCol="prediction", metricName="silhouette"
    )
    scores = []
    best_model = None
    best_score = float("-inf")
    for k in candidate_k:
        estimator = KMeans(
            k=k,
            seed=42,
            maxIter=100,
            initMode="k-means||",
            featuresCol="scaled_features",
            predictionCol="prediction",
        )
        model = estimator.fit(scaled)
        score = float(evaluator.evaluate(model.transform(scaled)))
        scores.append({"k": k, "silhouette": score})
        if score > best_score:
            best_model = model
            best_score = score
    if best_model is None:
        scaled.unpersist()
        raise RuntimeError("No K-Means candidate model was fitted.")
    pipeline_model = PipelineModel(stages=[assembler, scaler_model, best_model])
    scored = pipeline_model.transform(client_features)
    details = {
        "selected_k": best_model.getK(),
        "silhouette": best_score,
        "candidate_scores": scores,
        "client_count": client_features.count(),
    }
    scaled.unpersist()
    return pipeline_model, scored, details
