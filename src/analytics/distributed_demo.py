"""Small RDD, DataFrame and Spark SQL demonstrations over processed readings."""

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F


def rdd_top_clients_by_energy(readings: DataFrame, limit: int = 10) -> list[dict]:
    """Use RDD map, filter and reduceByKey without collecting source records."""
    if limit < 1:
        raise ValueError("limit must be at least 1")
    totals = (
        readings.select("client_id", "kwh")
        # A bounded worker count avoids spawning one Python process per CPU on
        # Windows laptops while the distributed reduce still combines client
        # keys across the full dataset.
        .rdd.coalesce(4)
        .filter(lambda row: row["client_id"] is not None and row["kwh"] is not None)
        .map(lambda row: (row["client_id"], float(row["kwh"])))
        .reduceByKey(lambda left, right: left + right)
    )
    return [
        {"client_id": client_id, "total_kwh": total_kwh}
        for client_id, total_kwh in totals.takeOrdered(limit, key=lambda item: -item[1])
    ]


def spark_sql_summary(spark: SparkSession, readings: DataFrame) -> dict:
    """Register a temporary view and run compact SQL aggregate queries."""
    readings.createOrReplaceTempView("powerpulse_readings")
    overall = spark.sql(
        """
        SELECT SUM(kwh) AS total_kwh,
               AVG(kw) AS average_kw,
               MAX(kw) AS maximum_client_kw,
               COUNT(DISTINCT client_id) AS client_count,
               COUNT(*) AS reading_count
        FROM powerpulse_readings
        """
    ).first()
    yearly = [
        row.asDict()
        for row in spark.sql(
            """
            SELECT year, SUM(kwh) AS total_kwh, AVG(kw) AS average_kw
            FROM powerpulse_readings
            GROUP BY year
            ORDER BY year
            """
        ).collect()
    ]
    return {"overall": overall.asDict(), "yearly": yearly}
