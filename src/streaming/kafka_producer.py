"""Replay processed historical readings into Kafka as JSON messages."""

import argparse
import json
import logging
from pathlib import Path
import sys
import time
from datetime import datetime

from kafka import KafkaProducer
from kafka.errors import KafkaError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import KAFKA_BOOTSTRAP_SERVERS, KAFKA_TOPIC, PROCESSED_DATA_PATH  # noqa: E402
from src.spark.spark_session import create_spark_session  # noqa: E402

LOGGER = logging.getLogger(__name__)


def replay_readings(bootstrap_servers=None, topic=None, limit=None, delay_seconds=0.0):
    """Publish readings in timestamp order, optionally limiting replay volume."""
    servers = bootstrap_servers or KAFKA_BOOTSTRAP_SERVERS
    topic = topic or KAFKA_TOPIC
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    if delay_seconds < 0:
        raise ValueError("delay_seconds cannot be negative")

    producer = KafkaProducer(
        bootstrap_servers=servers,
        value_serializer=lambda value: json.dumps(value, separators=(",", ":")).encode("utf-8"),
        retries=5,
        acks="all",
    )
    spark = None
    sent = 0
    try:
        spark = create_spark_session("PowerPulse-Kafka-Historical-Replay")
        readings = spark.read.parquet(str(PROCESSED_DATA_PATH)).select(
            "timestamp", "client_id", "kw", "kwh"
        ).orderBy("timestamp", "client_id")
        if limit is not None:
            readings = readings.limit(limit)
        for row in readings.toLocalIterator():
            timestamp = row.timestamp
            if isinstance(timestamp, datetime):
                timestamp = timestamp.isoformat(sep=" ")
            payload = {
                "timestamp": str(timestamp),
                "client_id": row.client_id,
                "kw": float(row.kw),
                "kwh": float(row.kwh),
            }
            producer.send(topic, value=payload)
            sent += 1
            if delay_seconds:
                time.sleep(delay_seconds)
            if sent % 10000 == 0:
                producer.flush()
                LOGGER.info("Published %s historical readings", sent)
        producer.flush(timeout=60)
        LOGGER.info("Replay complete: published %s messages to %s", sent, topic)
        return sent
    except KafkaError:
        LOGGER.exception("Kafka publish failed after %s messages", sent)
        raise
    finally:
        producer.close(timeout=10)
        if spark is not None:
            spark.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-servers", default=KAFKA_BOOTSTRAP_SERVERS)
    parser.add_argument("--topic", default=KAFKA_TOPIC)
    parser.add_argument("--limit", type=int, help="Maximum number of readings to replay")
    parser.add_argument("--delay-seconds", type=float, default=0.0)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    replay_readings(args.bootstrap_servers, args.topic, args.limit, args.delay_seconds)


if __name__ == "__main__":
    main()
