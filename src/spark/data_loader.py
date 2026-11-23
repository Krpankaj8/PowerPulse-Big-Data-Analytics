"""Load the original UCI electricity file as a Spark DataFrame."""

import csv
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import StringType, StructField, StructType


def load_uci_wide_dataset(spark: SparkSession, dataset_path: Path | str) -> DataFrame:
    """Read the semicolon-delimited UCI TXT file without guessing value types.

    Client readings remain strings at this stage so decimal commas and malformed
    values can be handled explicitly by the preprocessing module.
    """
    path = Path(dataset_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"UCI dataset file does not exist: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as source_file:
        header = next(csv.reader(source_file, delimiter=";", quotechar='"'), None)
    if not header or len(header) < 2:
        raise ValueError(f"No valid UCI header found in dataset: {path}")

    column_names = [header[0].strip()] + [name.strip() for name in header[1:]]
    if any(not name for name in column_names[1:]) or len(set(column_names)) != len(column_names):
        raise ValueError("The UCI header contains empty or duplicate client identifiers.")
    schema = StructType([StructField(name, StringType(), nullable=True) for name in column_names])

    frame = (
        spark.read.format("csv")
        .schema(schema)
        .option("header", "true")
        .option("enforceSchema", "false")
        .option("delimiter", ";")
        .option("quote", '"')
        .option("escape", '"')
        .option("encoding", "UTF-8")
        .option("inferSchema", "false")
        .option("mode", "PERMISSIVE")
        .load(str(path))
    )
    frame = frame.withColumnRenamed("", "timestamp_raw")
    if len(frame.columns) < 2:
        raise ValueError(
            "Expected a timestamp column and at least one client column in "
            f"the UCI file; found {len(frame.columns)} column(s)."
        )
    return frame
