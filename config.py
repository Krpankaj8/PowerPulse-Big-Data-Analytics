"""Central, environment-driven configuration for PowerPulse."""

from pathlib import Path
import os

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env", override=False)


def _project_path(variable: str, default: str) -> Path:
    """Resolve a configured path relative to the project unless it is absolute."""
    value = Path(os.getenv(variable, default)).expanduser()
    return value if value.is_absolute() else (PROJECT_ROOT / value).resolve()


DATASET_PATH = _project_path("POWERPULSE_DATASET_PATH", "data/raw/LD2011_2014.txt")
PROCESSED_DATA_PATH = _project_path("POWERPULSE_PROCESSED_DATA_PATH", "data/processed/readings")
OUTPUTS_PATH = _project_path("POWERPULSE_OUTPUTS_PATH", "outputs")
MODELS_PATH = _project_path("POWERPULSE_MODELS_PATH", "models")
LOGS_PATH = _project_path("POWERPULSE_LOGS_PATH", "logs")

SPARK_MASTER = os.getenv("POWERPULSE_SPARK_MASTER", "local[*]")
SPARK_APP_NAME = os.getenv("POWERPULSE_SPARK_APP_NAME", "PowerPulse")
SPARK_TIMEZONE = os.getenv("POWERPULSE_SPARK_TIMEZONE", "UTC")
KAFKA_BOOTSTRAP_SERVERS = os.getenv("POWERPULSE_KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC = os.getenv("POWERPULSE_KAFKA_TOPIC", "powerpulse-readings")
