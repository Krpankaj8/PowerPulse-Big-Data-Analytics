"""Small, explicit fixtures for UCI value parsing and interval conversion."""

import unittest

from pyspark.sql.types import TimestampNTZType

from src.spark.preprocessing import add_calendar_and_energy_features, unpivot_readings
from src.spark.spark_session import create_spark_session


class PreprocessingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.spark = create_spark_session("PowerPulse-Preprocessing-Unit-Tests")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.spark.stop()

    def test_decimal_comma_zero_and_calendar_features(self) -> None:
        wide = self.spark.createDataFrame(
            [
                ("2011-01-01 00:15:00", "71,7703349282297", "0"),
                ("2011-01-03 01:00:00", "12,5", "2,0"),
            ],
            ["", "MT_001", "MT_002"],
        )

        result = add_calendar_and_energy_features(unpivot_readings(wide))
        values = {
            (row["client_id"], row["date"].isoformat()): row
            for row in result.collect()
        }
        saturday_reading = values[("MT_001", "2011-01-01")]
        zero_reading = values[("MT_002", "2011-01-01")]
        monday_reading = values[("MT_001", "2011-01-03")]

        self.assertIsInstance(result.schema["timestamp"].dataType, TimestampNTZType)
        self.assertEqual(
            saturday_reading["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
            "2011-01-01 00:15:00",
        )
        self.assertAlmostEqual(saturday_reading["kw"], 71.7703349282297)
        self.assertAlmostEqual(saturday_reading["kwh"], 71.7703349282297 * 0.25)
        self.assertEqual(zero_reading["kw"], 0.0)
        self.assertEqual(zero_reading["kwh"], 0.0)
        self.assertTrue(saturday_reading["is_weekend"])
        self.assertFalse(monday_reading["is_weekend"])
        self.assertEqual(monday_reading["hour"], 1)
        self.assertEqual(monday_reading["month"], 1)


if __name__ == "__main__":
    unittest.main()
