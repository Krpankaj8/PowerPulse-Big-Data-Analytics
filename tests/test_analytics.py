"""Small test-only Spark rows for the distributed analytics helpers."""

from datetime import datetime
import unittest

from pyspark.sql.types import (
    BooleanType,
    DateType,
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
    TimestampNTZType,
)

from src.analytics.anomaly_detection import detect_client_zscore_anomalies
from src.analytics.client_analysis import client_profiles
from src.analytics.consumption_analysis import summarize_consumption
from src.analytics.distributed_demo import rdd_top_clients_by_energy, spark_sql_summary
from src.analytics.peak_analysis import peak_summary
from src.analytics.trend_analysis import daily_trend, weekday_weekend_trend
from src.spark.spark_session import create_spark_session


class AnalyticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.spark = create_spark_session("PowerPulse-Analytics-Unit-Tests")
        schema = StructType(
            [
                StructField("timestamp", TimestampNTZType(), False),
                StructField("client_id", StringType(), False),
                StructField("kw", DoubleType(), False),
                StructField("kwh", DoubleType(), False),
                StructField("hour", IntegerType(), False),
                StructField("day_of_week", IntegerType(), False),
                StructField("month", IntegerType(), False),
                StructField("is_weekend", BooleanType(), False),
                StructField("year", IntegerType(), False),
                StructField("date", DateType(), False),
            ]
        )
        samples = []
        for timestamp_text, day, is_weekend, client_values in [
            ("2011-01-01 00:15:00", 7, True, (10.0, 5.0)),
            ("2011-01-01 00:30:00", 7, True, (10.0, 5.0)),
            ("2011-01-03 00:45:00", 2, False, (10.0, 5.0)),
            ("2011-01-03 01:00:00", 2, False, (100.0, 5.0)),
        ]:
            timestamp = datetime.fromisoformat(timestamp_text)
            for client_id, kw in zip(("MT_001", "MT_002"), client_values):
                samples.append(
                    (
                        timestamp,
                        client_id,
                        kw,
                        kw * 0.25,
                        timestamp.hour,
                        day,
                        1,
                        is_weekend,
                        2011,
                        timestamp.date(),
                    )
                )
        cls.readings = cls.spark.createDataFrame(samples, schema=schema)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.spark.stop()

    def test_overall_trends_peaks_and_client_profiles(self) -> None:
        overall = summarize_consumption(self.readings)
        self.assertEqual(overall["number_of_readings"], 8)
        self.assertEqual(overall["number_of_clients"], 2)
        self.assertAlmostEqual(overall["total_energy_kwh"], 37.5)

        daily = {row["date"].isoformat(): row for row in daily_trend(self.readings).collect()}
        self.assertEqual(len(daily), 2)
        weekends = {row["is_weekend"]: row for row in weekday_weekend_trend(self.readings).collect()}
        self.assertEqual(weekends[True]["reading_count"], 4)
        self.assertEqual(weekends[False]["reading_count"], 4)

        peaks, by_hour = peak_summary(self.readings, top_n=2)
        self.assertAlmostEqual(peaks["maximum_aggregate_demand_kw"], 105.0)
        self.assertEqual(len(peaks["top_demand_timestamps"]), 2)
        self.assertEqual(by_hour.count(), 2)

        profiles = client_profiles(self.readings)
        top = profiles.orderBy("total_kwh", ascending=False).first()
        self.assertEqual(top["client_id"], "MT_001")

    def test_rdd_sql_and_client_relative_anomalies(self) -> None:
        rdd_top = rdd_top_clients_by_energy(self.readings, limit=1)
        self.assertEqual(rdd_top[0]["client_id"], "MT_001")

        sql_results = spark_sql_summary(self.spark, self.readings)
        self.assertEqual(sql_results["overall"]["reading_count"], 8)
        self.assertEqual(len(sql_results["yearly"]), 1)

        anomaly_rows = detect_client_zscore_anomalies(self.readings, threshold=1.5).collect()
        self.assertEqual(len(anomaly_rows), 1)
        self.assertEqual(anomaly_rows[0]["client_id"], "MT_001")
        self.assertEqual(anomaly_rows[0]["kw"], 100.0)


if __name__ == "__main__":
    unittest.main()
