param(
    [string]$BootstrapServers = "localhost:9092",
    [string]$Topic = "powerpulse-readings",
    [string]$Window = "1 minute",
    [string]$Checkpoint,
    [switch]$Once
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$sparkSubmit = Join-Path $env:SPARK_HOME "bin\spark-submit.cmd"
$python = Join-Path $root ".venv\Scripts\python.exe"
$connector = "org.apache.spark:spark-sql-kafka-0-10_2.13:4.2.0"
$submitArguments = @(
    "--packages", $connector,
    "--conf", "spark.pyspark.python=$python",
    (Join-Path $root "src\streaming\spark_consumer.py"),
    "--bootstrap-servers", $BootstrapServers,
    "--topic", $Topic,
    "--window", $Window
)
if ($Checkpoint) { $submitArguments += @("--checkpoint", $Checkpoint) }
if ($Once) { $submitArguments += "--once" }
& $sparkSubmit @submitArguments
if ($LASTEXITCODE -ne 0) { throw "Structured Streaming exited with code $LASTEXITCODE." }
