"""Transform UCI's wide power readings into a long, calendar-enriched table."""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def unpivot_readings(wide: DataFrame) -> DataFrame:
    """Convert each timestamp/client cell to one typed long-format reading.

    The first source column is the timestamp even though UCI leaves its header
    blank. All other columns are client identifiers. Values use decimal commas
    in the original file, so commas are changed to decimal points before casting.
    """
    if len(wide.columns) < 2:
        raise ValueError("A wide dataset must have a timestamp and client columns.")

    timestamp_source = wide.columns[0]
    client_columns = wide.columns[1:]
    cells = wide.unpivot(
        ids=[timestamp_source],
        values=client_columns,
        variableColumnName="client_id",
        valueColumnName="reading_raw",
    )
    return (
        cells.withColumn(
            "timestamp",
            F.to_timestamp_ntz(
                F.trim(F.col(timestamp_source)),
                F.lit("yyyy-MM-dd HH:mm:ss"),
            ),
        )
        .withColumn(
            "kw",
            F.regexp_replace(F.trim(F.col("reading_raw")), ",", ".").cast("double"),
        )
        .select("timestamp", "client_id", "kw")
    )


def add_calendar_and_energy_features(readings: DataFrame) -> DataFrame:
    """Add interval energy and calendar columns required by downstream modules.

    UCI readings are average kW over 15 minutes, so interval energy is kW / 4.
    Spark's ``dayofweek`` convention is used: Sunday=1 through Saturday=7.
    """
    required = {"timestamp", "client_id", "kw"}
    missing = required.difference(readings.columns)
    if missing:
        raise ValueError(f"Readings DataFrame is missing required columns: {sorted(missing)}")

    result = (
        readings.withColumn("kwh", F.col("kw") * F.lit(0.25))
        .withColumn("hour", F.hour("timestamp"))
        .withColumn("day_of_week", F.dayofweek("timestamp"))
        .withColumn("month", F.month("timestamp"))
        .withColumn("is_weekend", F.dayofweek("timestamp").isin(1, 7))
        .withColumn("year", F.year("timestamp"))
        .withColumn("date", F.to_date("timestamp"))
    )
    return result.select(
        "timestamp",
        "client_id",
        "kw",
        "kwh",
        "hour",
        "day_of_week",
        "month",
        "is_weekend",
        "year",
        "date",
    )
