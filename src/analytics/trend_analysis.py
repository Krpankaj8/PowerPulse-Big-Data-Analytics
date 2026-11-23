"""Daily, monthly, hourly, annual and calendar-based trends."""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def daily_trend(readings: DataFrame) -> DataFrame:
    return readings.groupBy("date").agg(
        F.sum("kwh").alias("total_kwh"),
        F.avg("kw").alias("average_kw"),
    ).orderBy("date")


def monthly_trend(readings: DataFrame) -> DataFrame:
    return readings.groupBy("year", "month").agg(
        F.sum("kwh").alias("total_kwh"),
        F.avg("kw").alias("average_kw"),
    ).orderBy("year", "month")


def yearly_trend(readings: DataFrame) -> DataFrame:
    return readings.groupBy("year").agg(
        F.sum("kwh").alias("total_kwh"),
        F.avg("kw").alias("average_kw"),
        F.max("kw").alias("maximum_client_kw"),
    ).orderBy("year")


def hourly_trend(readings: DataFrame) -> DataFrame:
    return readings.groupBy("hour").agg(
        F.avg("kw").alias("average_kw"),
        F.sum("kwh").alias("total_kwh"),
    ).orderBy("hour")


def weekday_weekend_trend(readings: DataFrame) -> DataFrame:
    return readings.groupBy("is_weekend").agg(
        F.count(F.lit(1)).alias("reading_count"),
        F.avg("kw").alias("average_kw"),
        F.sum("kwh").alias("total_kwh"),
    ).orderBy("is_weekend")
