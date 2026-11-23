"""Aggregate Kafka JSON readings with Spark Structured Streaming."""

import argparse
import logging
from pathlib import Path
import sys

from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, StringType, StructField, StructType, TimestampType

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import KAFKA_BOOTSTRAP_SERVERS, KAFKA_TOPIC, OUTPUTS_PATH  # noqa: E402
from src.spark.spark_session import create_spark_session  # noqa: E402

MESSAGE_SCHEMA = StructType(
    [
        StructField("timestamp", TimestampType(), True),
        StructField("client_id", StringType(), True),
        StructField("kw", DoubleType(), True),
        StructField("kwh", DoubleType(), True),
    ]
)


def run_consumer(bootstrap_servers=None, topic=None, checkpoint=None, window="1 minute", once=False):
    """Read Kafka and persist event-time window aggregates as Parquet."""
    servers = bootstrap_servers or KAFKA_BOOTSTRAP_SERVERS
    topic = topic or KAFKA_TOPIC
    output = OUTPUTS_PATH / "streaming" / "microbatch_aggregates"
    checkpoint = checkpoint or str(OUTPUTS_PATH / "streaming" / "checkpoint_microbatch")
    output.mkdir(parents=True, exist_ok=True)
    spark = create_spark_session("PowerPulse-Structured-Streaming")
    try:
        raw = (
            spark.readStream.format("kafka")
            .option("kafka.bootstrap.servers", servers)
            .option("subscribe", topic)
            .option("startingOffsets", "earliest")
            .option("failOnDataLoss", "false")
            .load()
        )
        parsed = raw.select(F.from_json(F.col("value").cast("string"), MESSAGE_SCHEMA).alias("reading"))
        valid = parsed.select("reading.*").where(
            F.col("timestamp").isNotNull()
            & F.col("client_id").isNotNull()
            & F.col("kw").isNotNull()
            & F.col("kwh").isNotNull()
        )

        def write_microbatch(batch, batch_id):
            # Kafka batches contain disjoint offsets. Aggregating each batch
            # avoids waiting for a future watermark to close old replay windows.
            (
                batch.select("timestamp", "client_id", "kw", "kwh")
                .orderBy(F.desc("timestamp"))
                .limit(100)
                .write.mode("overwrite")
                .parquet(str(output.parent / "latest_readings"))
            )
            results = (
                batch.groupBy(F.window("timestamp", window))
                .agg(
                    F.sum("kw").alias("sum_kw_readings"),
                    F.sum("kwh").alias("total_kwh"),
                    F.count("*").alias("reading_count"),
                )
                .select(
                    F.col("window.start").alias("window_start"),
                    F.col("window.end").alias("window_end"),
                    "sum_kw_readings",
                    "total_kwh",
                    "reading_count",
                )
                .withColumn("batch_id", F.lit(batch_id))
            )
            results.write.mode("append").parquet(str(output))

        writer = valid.writeStream.foreachBatch(write_microbatch).option("checkpointLocation", checkpoint)
        if once:
            writer = writer.trigger(availableNow=True)
        writer.start().awaitTermination()
    except Exception:
        logging.exception("Structured Streaming consumer failed")
        raise
    finally:
        spark.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-servers", default=KAFKA_BOOTSTRAP_SERVERS)
    parser.add_argument("--topic", default=KAFKA_TOPIC)
    parser.add_argument("--checkpoint")
    parser.add_argument("--window", default="1 minute", help='Spark window duration, e.g. "1 minute"')
    parser.add_argument("--once", action="store_true", help="Process available Kafka records then exit")
    args = parser.parse_args()
    run_consumer(args.bootstrap_servers, args.topic, args.checkpoint, args.window, args.once)


if __name__ == "__main__":
    main()
