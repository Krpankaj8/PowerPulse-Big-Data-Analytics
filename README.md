# PowerPulse

PowerPulse is a local electricity analytics and forecasting project. It turns the UCI ElectricityLoadDiagrams20112014 15-minute client readings into a Parquet dataset, calculates distributed analytics with Apache Spark, trains Spark MLlib models, builds a client similarity graph with Scala GraphX, optionally replays historical data through Kafka, and presents the results in a Streamlit dashboard.

The Kafka workflow replays historical records from the dataset. It does not connect to live meters. The GraphX network represents similar customer usage patterns, not physical electrical connections.

## Contents

- [What the project does](#what-the-project-does)
- [How the pieces fit together](#how-the-pieces-fit-together)
- [Project layout](#project-layout)
- [Requirements](#requirements)
- [Install and configure on Windows](#install-and-configure-on-windows)
- [Run the project from start to finish](#run-the-project-from-start-to-finish)
- [Run each pipeline](#run-each-pipeline)
- [Start the dashboard](#start-the-dashboard)
- [Configuration reference](#configuration-reference)
- [Outputs and data formats](#outputs-and-data-formats)
- [Tests](#tests)
- [Troubleshooting](#troubleshooting)

## What the project does

1. Reads the original semicolon-delimited UCI text file with Spark.
2. Converts its wide table (one column per client) into one row per timestamp and client.
3. Converts decimal commas to decimal points, preserves zero readings, derives 15-minute interval energy, and adds calendar fields.
4. Computes overall, time-based, client, peak-demand, and anomaly summaries using Spark DataFrames, Spark SQL, and an RDD example.
5. Builds time-based system-demand features and trains regression, classification, and client-clustering models with Spark MLlib.
6. Builds a cosine-similarity graph from client hourly load profiles using Spark GraphX.
7. Optionally publishes historical readings to Kafka and aggregates them with Spark Structured Streaming.
8. Loads the generated compact results in a seven-page Streamlit dashboard.

## How the pieces fit together

```text
data/raw/LD2011_2014.txt
          |
          v
 scripts/process_dataset.py
          |
          v
data/processed/readings  (Parquet, shared input to all downstream jobs)
       |                |                |
       v                v                v
run_analytics.py   train_models.py   GraphX build/run
       |                |                |
       +----------------+----------------+
                        v
                  outputs/ and models/
                        |
                        v
                 streamlit run app.py

Optional historical stream:
processed Parquet -> Kafka producer -> Kafka topic -> Spark consumer -> outputs/streaming
```

The dashboard reads the compact generated results; it does not load all 51 million readings into Pandas. Spark performs the large reads and aggregations. The dashboard starts or reuses a local Spark session when it needs to open Parquet results.

## Project layout

| Path | Purpose |
| --- | --- |
| `app.py` | Streamlit entry point and dashboard page renderers. |
| `config.py` | Reads `.env` and defines paths and Spark/Kafka defaults. |
| `.env.example` | Template for local configuration. Copy it to `.env`. |
| `requirements.txt` | Python dependencies. |
| `data/raw/` | Original UCI TXT input. The expected default file is `LD2011_2014.txt`. |
| `data/processed/` | Long-format Parquet output from preprocessing. |
| `src/spark/` | UCI loading, reading transformations, quality reports, and Spark session setup. |
| `src/analytics/` | Consumption, trend, peak, client, anomaly, RDD, and Spark SQL analytics. |
| `src/ml/` | Feature engineering, MLlib model pipelines, clustering, evaluation, and comparison. |
| `src/streaming/` | Kafka historical producer and Spark Structured Streaming consumer. |
| `scala/graphx-client-similarity/` | Scala GraphX application and optional SBT build definition. |
| `scripts/` | PowerShell/Python commands for validation, pipelines, Kafka, and GraphX. |
| `outputs/` | Generated quality, analytics, ML, graph, and streaming results. |
| `models/` | Saved Spark MLlib models. |
| `logs/` | Local Kafka configuration and KRaft broker data used by the startup script. |
| `tests/` | Small Spark fixture tests for preprocessing, analytics, and ML utilities. |
| `notebooks/` | Workspace for exploratory notebooks. |

### Main implementation modules

| Module | Responsibility |
| --- | --- |
| `src/spark/data_loader.py` | Reads the semicolon-delimited UCI file with an explicit string schema. |
| `src/spark/preprocessing.py` | Unpivots client readings and adds kWh/calendar fields. |
| `src/spark/quality_report.py` | Calculates distributed null, zero, client, row, and timestamp statistics. |
| `src/spark/spark_session.py` | Applies shared Spark master, timezone, Python worker, and session settings. |
| `src/analytics/consumption_analysis.py` | Overall power, energy, client, and reading totals. |
| `src/analytics/trend_analysis.py` | Daily, monthly, yearly, hourly, and weekday/weekend summaries. |
| `src/analytics/peak_analysis.py` | Timestamp-level aggregate demand and peak intervals. |
| `src/analytics/client_analysis.py` | Client-level profiles and top/bottom ranking. |
| `src/analytics/anomaly_detection.py` | Client-relative z-score anomaly screening. |
| `src/analytics/distributed_demo.py` | RDD transformations and Spark SQL summaries. |
| `src/ml/features.py` | System-demand features, chronological split, and training class weights. |
| `src/ml/linear_regression.py` | Linear regression pipeline for continuous demand. |
| `src/ml/logistic_regression.py` | Logistic regression pipeline for high-demand classification. |
| `src/ml/decision_tree.py` | Decision tree pipeline for high-demand classification. |
| `src/ml/naive_bayes.py` | Multinomial Naive Bayes pipeline for high-demand classification. |
| `src/ml/clustering.py` | Scaled client profile features and silhouette-selected K-Means. |
| `src/ml/evaluation.py` | Regression and binary classification metrics. |
| `src/ml/model_comparison.py` | Ranks trained models for comparison. |
| `src/streaming/kafka_producer.py` | Reads processed Parquet and publishes historical JSON messages to Kafka. |
| `src/streaming/spark_consumer.py` | Parses Kafka messages and writes event-time window aggregates. |
| `ClientSimilarityGraph.scala` | Builds and exports the GraphX client similarity network. |

Generated directories (`.venv`, `data/processed`, `outputs`, `models`, `logs`, and build artifacts) are ignored by Git. Keep a copy of anything in them that you need to preserve before cleaning or regenerating it.

## Requirements

The Windows setup used to develop this project has these compatible versions:

| Component | Project version / requirement | Used for |
| --- | --- | --- |
| Windows PowerShell | Windows PowerShell 5.1 or PowerShell 7 | Commands in this guide. |
| Python | 3.10 or newer; project environment was Python 3.11.9 | App, PySpark, producer. |
| PySpark | 4.2.0 | Distributed processing and MLlib. |
| Java | Java 17 | Spark, Kafka, and Scala compilation. |
| Apache Spark | 4.2.0; set `SPARK_HOME` | Spark scripts, `spark-submit`, and Spark JARs for GraphX. |
| Kafka | 4.2.1 KRaft distribution for the supplied startup script | Optional broker and topic. |
| Scala | 2.13.18 | GraphX source; the Spark distribution supplies compiler/runtime JARs. |

Python package requirements are in `requirements.txt`. The GraphX PowerShell build uses the Scala compiler and Spark JARs under `SPARK_HOME`; it does not require SBT. The optional SBT project declares Scala 2.13.18, Spark SQL/GraphX 4.2.0, and SBT 1.10.7.

The default Kafka script path is `C:\kafka\kafka_2.13-4.2.1\kafka_2.13-4.2.1`. Install Kafka there or pass your actual Kafka directory as `-KafkaHome` to each `kafka_startup.ps1` command.

## Install and configure on Windows

Run these commands from the repository root, the directory containing `app.py` and `README.md`.

### 1. Install and verify prerequisites

Install Python 3.10+, Java 17, Apache Spark 4.2.0, and (if you want the streaming feature) Kafka 4.2.1. Configure `JAVA_HOME` and `SPARK_HOME` to point to the Java and Spark installations. Make sure their `bin` directories are on `PATH` so `java`, `jar`, `spark-submit`, and Spark scripts can be found.

Check the commands and environment variables in PowerShell:

```powershell
Get-Command python, java, spark-submit
python --version
java -version
$env:JAVA_HOME
$env:SPARK_HOME
```

Use Python 3.10+ explicitly if the `python` command resolves to an older installation. For example, if Python 3.11 is installed under your user profile:

```powershell
& "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe" --version
```

### 2. Create the Python environment

Create the virtual environment with the same Python installation you intend to use for this project:

```powershell
& "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe" -m venv .venv
```

If Python is installed elsewhere, replace the path above. Activate the environment and install dependencies:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, skip activation and use the environment executables directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\streamlit.exe run app.py
```

If `.venv` was created with a Python installation that has since been uninstalled or moved, recreate it using the Python 3.10+ executable that is currently installed. The environment is generated and ignored by Git; recreating it removes packages inside `.venv`, so reinstall them with `pip install -r requirements.txt`.

### 3. Create local settings

Copy the example file once:

```powershell
Copy-Item .env.example .env
```

Edit `.env` only for paths or service settings that differ from the defaults. `.env` is ignored by Git and should remain local to your machine.

### Dataset source

The original data is distributed by the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014). Download `LD2011_2014.txt` and place it at `data/raw/LD2011_2014.txt`, or set `POWERPULSE_DATASET_PATH` to its location. The raw file is large (about 678 MB uncompressed). Cite the source as: Trindade, A. (2015), *ElectricityLoadDiagrams20112014*, UCI Machine Learning Repository, [doi:10.24432/C58C86](https://doi.org/10.24432/C58C86). The dataset is distributed under CC BY 4.0; retain attribution when sharing derived work.

### 4. Verify Spark

```powershell
python scripts\check_spark.py
```

The check starts a local Spark session, counts ten generated rows, prints the Spark version and master, and shuts the session down. Fix Java/Spark/Python path issues before running the full dataset pipeline.

## Run the project from start to finish

Run from the project root in this order. The analytics, ML, and GraphX outputs fill the non-streaming dashboard pages; Kafka is optional:

```powershell
python scripts\check_spark.py
python scripts\process_dataset.py
python scripts\run_analytics.py
python scripts\train_models.py
.\scripts\build_graphx.ps1
.\scripts\run_graphx.ps1 -SimilarityThreshold 0.995
streamlit run app.py
```

Processing the full data set can take a while and needs substantial disk space for Parquet outputs. Do not run all jobs simultaneously on a laptop. Run Kafka separately only if you want the Streaming Analytics page populated.

The processing and analytics jobs refuse to replace existing outputs unless you pass `--overwrite`. The ML command likewise protects existing model/output directories. GraphX protects its output unless `-Overwrite` is supplied. Back up any generated results you want to keep before using overwrite options.

## Run each pipeline

### Dataset preprocessing

The supplied UCI data is a wide semicolon-delimited TXT file: its first column contains timestamp labels and its remaining columns are client identifiers. Each client value is a decimal-comma average power reading in kW for a 15-minute interval.

Run:

```powershell
python scripts\process_dataset.py
```

The pipeline reads values as strings, parses timestamps as `timestamp_ntz` so the source's wall-clock labels are retained, changes decimal commas to decimal points, casts the measurements to doubles, and unpivots client columns to rows. It computes `kwh = kw * 0.25`, retains zero readings, and adds hour, day of week, month, weekend, year, and date columns. Spark day-of-week uses Sunday=1 through Saturday=7.

By default it writes:

- `data/processed/readings/` — Parquet rows with `timestamp`, `client_id`, `kw`, `kwh`, `hour`, `day_of_week`, `month`, `is_weekend`, `year`, and `date`.
- `outputs/data_quality.json` — row/client counts, null and zero counts, timestamp range, source metadata, and compact counts per client.

Change paths or intentionally regenerate results:

```powershell
python scripts\process_dataset.py --input data\raw\LD2011_2014.txt --output data\processed\readings --report outputs\data_quality.json --overwrite
```

### Distributed analytics

Run after preprocessing:

```powershell
python scripts\run_analytics.py
```

This job calculates total energy and power summaries, daily/monthly/yearly/hourly and weekday/weekend trends, peak system demand, client profiles, client-relative anomalies, top/bottom clients, an RDD `map`/`filter`/`reduceByKey` example, and Spark SQL summaries. It writes Parquet directories and a compact JSON summary below `outputs/analytics/`.

The default anomaly rule flags a reading when its absolute per-client z-score is greater than 3.0. These are statistical review flags; they do not diagnose equipment failure. To change the cutoff or output path:

```powershell
python scripts\run_analytics.py --anomaly-z-threshold 3.0 --output outputs\analytics
```

Use `--overwrite` only to replace existing analytics outputs.

### Spark MLlib models

Run after preprocessing:

```powershell
python scripts\train_models.py
```

The script first aggregates all clients into total system demand at each timestamp. It creates calendar features plus the previous 15-minute demand and previous 96th reading (nominally the same time on the previous day). Rows without both prior values are omitted. A chronological split keeps future timestamps out of training; default training fraction is 80%.

The job fits:

- Linear Regression for aggregate system demand in kW; metrics include MAE, RMSE, and R-squared.
- Logistic Regression, Decision Tree, and Multinomial Naive Bayes for high-demand classification. The high-demand cutoff is the configured training-only 90th percentile by default. Inverse-frequency class weights are calculated from training rows only.
- K-Means client profiles for k=2 through 6. Client statistics and 24 mean hourly loads are scaled, and the model with the best silhouette score is selected.

Predictions, metrics, model comparisons, cluster assignments, and saved models go to `outputs/ml/` and `models/ml/`. Change the input/output paths, split, or demand quantile with options:

```powershell
python scripts\train_models.py --train-fraction 0.8 --high-demand-quantile 0.9
```

Both values must be greater than 0.5 and less than 1.0. Use `--overwrite` to replace existing model and result directories.

### Scala Spark GraphX

Build the jar (requires `SPARK_HOME` and Java tools on `PATH`):

```powershell
.\scripts\build_graphx.ps1
```

Run it against the processed Parquet data:

```powershell
.\scripts\run_graphx.ps1 -SimilarityThreshold 0.995
```

For each client, the application calculates a 24-element profile of mean kW by hour. It connects pairs whose cosine similarity meets the threshold, then calculates degree, connected components, and PageRank. The threshold must be greater than 0 and at most 1. Outputs are CSV part files in `outputs/graphx/nodes/` and `outputs/graphx/edges/`, plus `outputs/graphx/graph_summary.json`. The default script threshold is 0.98; pass `-SimilarityThreshold` to change it. Existing output is protected; pass `-Overwrite` when replacement is intended.

### Kafka historical replay and Spark Structured Streaming

This optional workflow needs a working Kafka 4.2.1 KRaft install, Java, `SPARK_HOME`, the processed Parquet dataset, and local Maven access the first time Spark downloads its Kafka connector. The default Kafka home is `C:\kafka\kafka_2.13-4.2.1\kafka_2.13-4.2.1`; supply `-KafkaHome` if yours differs.

#### First-time Kafka storage setup

Only when starting a new local Kafka data directory, format storage once:

```powershell
.\scripts\kafka_startup.ps1 -Action format
```

Formatting initializes the KRaft storage. Do not repeat it for an already initialized `logs\kafka-kraft-data` directory; preserve that directory if you need its broker/topic data.

#### Start the broker and run a replay

Use four PowerShell windows total if the dashboard is already running: one for Streamlit, one for Kafka, one for topic creation and the producer, and one for the consumer.

**Window 1 — dashboard:**

```powershell
streamlit run app.py
```

**Window 2 — broker:** start this after any first-time storage setup. This command stays in the foreground; leave the window open:

```powershell
.\scripts\kafka_startup.ps1 -Action start
```

**Window 3 — topic and historical producer:** wait until the broker has started, then create the topic and replay the first 10,000 readings. These commands run one after the other in the same window:

```powershell
.\scripts\kafka_startup.ps1 -Action create-topic
python -m src.streaming.kafka_producer --limit 10000
```

To replay the full dataset, omit `--limit 10000`. To slow replay, add for example `--delay-seconds 0.01`. The producer reads from the processed Parquet dataset and publishes JSON messages with timestamp, client ID, kW, and kWh.

**Window 4 — Structured Streaming consumer:** after messages are available, process the current backlog and exit:

```powershell
.\scripts\run_streaming_consumer.ps1 -Once
```

The default consumer runs continuously if `-Once` is omitted. It groups readings into one-minute event-time windows, appends aggregates to `outputs/streaming/microbatch_aggregates/`, and overwrites the latest-reading snapshot under `outputs/streaming/latest_readings/`. The default checkpoint is `outputs/streaming/checkpoint_microbatch/`.

Spark Structured Streaming remembers consumed Kafka offsets in its checkpoint. Reusing a checkpoint continues from its saved offsets. For a fresh replay, either use a new topic or provide a new checkpoint path with `-Checkpoint`; use a new checkpoint if the aggregation/window definition changes. Do not delete checkpoint/output data unless you intend to reset that run.

The dashboard's Kafka status checks broker reachability. “Unavailable” means it could not connect to the configured broker; it is separate from whether streaming aggregate files already exist. A visible historical count can therefore remain while the broker is stopped.

## Start the dashboard

After generating the outputs you want to view, run from the project root:

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

Streamlit prints a local URL, normally `http://localhost:8501`. Open it in a browser. Keep the PowerShell window open while using the app. If the environment is not activated, use:

```powershell
.\.venv\Scripts\streamlit.exe run app.py
```

The sidebar contains these pages:

| Page | Data shown |
| --- | --- |
| Overview | Overall energy, mean power, peak demand, client/reading counts, daily energy trend. |
| Consumption Analytics | Daily, monthly, yearly, hourly, and weekday/weekend summaries. |
| Client Analytics | Client totals, distribution, selected-client comparisons, and profile rows. |
| Anomaly Detection | Client-relative z-score count, flagged clients, and sample flagged readings. |
| Machine Learning | Regression and classification comparisons, held-out predictions, client clusters. |
| Graph Analytics | Similarity graph metrics, degree distribution, PageRank, and selected links. |
| Streaming Analytics | Kafka broker status, processed window aggregates, and latest reading sample. |

If an output family has not been generated, its page will show a message directing you to the corresponding pipeline. Use the Streamlit menu's rerun control or refresh the browser after generating more outputs.

## Configuration reference

`config.py` loads `.env` from the repository root. Relative paths resolve from that root; absolute Windows paths are also supported. Environment variables already set in the shell take precedence over `.env` because dotenv does not override existing values.

| Variable | Default | Purpose |
| --- | --- | --- |
| `POWERPULSE_DATASET_PATH` | `data/raw/LD2011_2014.txt` | Original UCI TXT input. |
| `POWERPULSE_PROCESSED_DATA_PATH` | `data/processed/readings` | Preprocessed Parquet directory. |
| `POWERPULSE_OUTPUTS_PATH` | `outputs` | Root for reports, analytics, model outputs, graph, streaming. |
| `POWERPULSE_MODELS_PATH` | `models` | Root for saved Spark MLlib models. |
| `POWERPULSE_LOGS_PATH` | `logs` | Project log and Kafka KRaft data root. |
| `POWERPULSE_SPARK_MASTER` | `local[*]` | Spark master; defaults to local CPU cores. |
| `POWERPULSE_SPARK_APP_NAME` | `PowerPulse` | Default Spark application name. |
| `POWERPULSE_SPARK_TIMEZONE` | `UTC` | Spark SQL session timezone. Source labels are parsed into timezone-naive timestamps. |
| `POWERPULSE_KAFKA_BOOTSTRAP_SERVERS` | `localhost:9092` | Kafka broker address. |
| `POWERPULSE_KAFKA_TOPIC` | `powerpulse-readings` | Topic used by the producer, consumer, and dashboard. |

Some scripts also have command-line options for input/output locations and behavior. Run `python scripts\process_dataset.py --help`, `python scripts\run_analytics.py --help`, or `python scripts\train_models.py --help` to see them. PowerShell scripts accept parameters in their source declarations; see their usage in this guide or inspect the script's `param` block.

## Outputs and data formats

| Output | Format | Created by |
| --- | --- | --- |
| `outputs/data_quality.json` | JSON | `process_dataset.py` |
| `outputs/analytics/analytics_summary.json` | JSON | `run_analytics.py` |
| `outputs/analytics/daily_trend`, `monthly_trend`, `yearly_trend`, `hourly_trend`, `weekday_weekend_trend` | Spark Parquet directories | `run_analytics.py` |
| `outputs/analytics/aggregate_demand_by_hour`, `client_profiles`, `anomalies` | Spark Parquet directories | `run_analytics.py` |
| `outputs/ml/regression_metrics.json`, `classification_metrics.json`, `clustering_metrics.json`, `model_comparison.json` | JSON | `train_models.py` |
| `outputs/ml/linear_regression_predictions`, `classification_predictions`, `client_clusters`, `cluster_characteristics` | Spark Parquet directories | `train_models.py` |
| `models/ml/*` | Spark MLlib model directories | `train_models.py` |
| `outputs/graphx/graph_summary.json` | JSON | `run_graphx.ps1` |
| `outputs/graphx/nodes`, `outputs/graphx/edges` | CSV part files | `run_graphx.ps1` |
| `outputs/streaming/microbatch_aggregates` | Spark Parquet directory | `run_streaming_consumer.ps1` |
| `outputs/streaming/latest_readings` | Spark Parquet directory | `run_streaming_consumer.ps1` |
| `outputs/streaming/checkpoint_microbatch` | Spark checkpoint state | Structured Streaming consumer |

Spark Parquet output paths are directories, even when a job writes only a small table. Inspect them with Spark/Pandas or by opening their generated part files; do not assume the directory is a single `.parquet` file.

## Tests

The tests use small in-memory Spark fixtures and do not need the full UCI dataset or Kafka. From the activated environment, run:

```powershell
python -m unittest discover -s tests -v
```

They cover decimal-comma parsing and time features, aggregate/trend/peak/client/anomaly calculations, RDD and SQL helpers, chronological splitting, class weighting, and evaluation metrics. A full end-to-end data rebuild is separate from these unit tests.

## Troubleshooting

### `KeyboardInterrupt` during Python or Streamlit startup

`KeyboardInterrupt` means the process received an interrupt (often Ctrl+C); it is not by itself a Streamlit exception. If startup is merely slow, allow it to finish. If the virtual environment points to a Python executable that was moved or uninstalled, install Python 3.10+ and recreate `.venv` with that interpreter, then reinstall `requirements.txt` as described above.

### Streamlit opens, but Kafka says “Unavailable”

The dashboard is running, but it cannot reach the broker at `localhost:9092` (or the configured address). Start Kafka and leave its PowerShell window open. Create the topic, replay readings, and run the consumer using the four-window sequence in [Kafka historical replay and Spark Structured Streaming](#kafka-historical-replay-and-spark-structured-streaming). Topic creation or consumer execution will not start the broker automatically.

### Kafka script says the Kafka path was not found

The startup script defaults to `C:\kafka\kafka_2.13-4.2.1\kafka_2.13-4.2.1`. Pass the actual install path, for example:

```powershell
.\scripts\kafka_startup.ps1 -Action start -KafkaHome 'D:\tools\kafka_2.13-4.2.1'
```

Use the same `-KafkaHome` for `format` and `create-topic`.

### `JAVA_HOME`, `SPARK_HOME`, or `spark-submit` is missing

Set `JAVA_HOME` to the Java 17 installation and `SPARK_HOME` to the Spark 4.2.0 installation, then add their `bin` directories to `PATH`. Open a new PowerShell window after changing system/user environment variables and re-check with `Get-Command java, spark-submit`.

### Output already exists / overwrite protection

Pipelines refuse to replace existing output by default. Decide whether the current generated data should be kept, then either choose a new output path or use the relevant `--overwrite`/`-Overwrite` option. Overwrite replaces generated output and may remove results that the dashboard currently displays.

### Dashboard page says data is missing

Run the pipeline for that page after preprocessing: analytics (`run_analytics.py`), ML (`train_models.py`), GraphX (build and run scripts), or streaming (start Kafka, produce, consume). Confirm that the expected directory exists beneath the configured `outputs` root.

### Spark cannot download the Kafka connector

The first consumer invocation may download `org.apache.spark:spark-sql-kafka-0-10_2.13:4.2.0` from Maven. Allow network access for that first run or resolve the connector in an environment that can access Maven repositories. A broker being available does not make the connector dependency available automatically.

### Large data jobs are slow or run out of memory

The supplied file expands to roughly 52 million readings. Run one Spark pipeline at a time, keep adequate free disk space for Parquet, close unrelated heavy processes, and avoid loading raw readings into Pandas. The dashboard caps many Parquet reads to compact samples. For local development, start with a smaller copy of the source file or a separate test fixture rather than replacing production outputs.

## Reproducibility notes

The recorded UCI run produced 51,894,720 client readings for 370 clients and 140,256 timestamps, from 2011-01-01 00:15 through 2015-01-01 00:00. The quality report recorded zero null power values, zero invalid timestamps, and 10,457,342 zero-power readings. Results are computed from the dataset, not hardcoded into the dashboard.

Analytics from the recorded local run included total energy of 6,857,008,632.74 kWh and peak aggregate demand of 452,652.60 kW at 2013-07-09 17:00. Trained metrics and GraphX counts are stored in the generated JSON files and can vary if configuration, input data, or library versions change. Streaming output is a historical replay, not live telemetry.
