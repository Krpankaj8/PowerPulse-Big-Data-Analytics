param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("format", "start", "create-topic")]
    [string]$Action,
    [string]$KafkaHome = "C:\kafka\kafka_2.13-4.2.1\kafka_2.13-4.2.1",
    [string]$Topic = "powerpulse-readings",
    [string]$BootstrapServers = "localhost:9092"
)

$ErrorActionPreference = "Stop"
$javaClasspath = "$(Join-Path $KafkaHome 'libs\*');$(Join-Path $KafkaHome 'config')"
$brokerConfig = Join-Path $KafkaHome "config\server.properties"
$projectRoot = Split-Path -Parent $PSScriptRoot
$runtimeConfig = Join-Path $projectRoot "logs\kafka-server.properties"
$dataPath = Join-Path $projectRoot "logs\kafka-kraft-data"

foreach ($path in @($KafkaHome, $brokerConfig)) {
    if (-not (Test-Path -LiteralPath $path)) { throw "Kafka path not found: $path" }
}

$configText = Get-Content -LiteralPath $brokerConfig -Raw
if ($configText -notmatch '(?m)^process\.roles\s*=') {
    throw "Kafka config does not declare KRaft process.roles; choose a KRaft config."
}
$configText = [regex]::Replace($configText, '(?m)^log\.dirs\s*=.*$', "log.dirs=$($dataPath.Replace('\', '/'))")
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText($runtimeConfig, $configText, $utf8NoBom)

switch ($Action) {
    "format" {
        $storageOutput = & java -cp $javaClasspath kafka.tools.StorageTool random-uuid
        if ($LASTEXITCODE -ne 0 -or -not $storageOutput) { throw "Kafka StorageTool random-uuid failed." }
        $clusterId = ($storageOutput | Select-Object -Last 1).Trim()
        if (-not $clusterId) { throw "Kafka did not return a KRaft cluster ID." }
        & java -cp $javaClasspath kafka.tools.StorageTool format --standalone -t $clusterId -c $runtimeConfig
        if ($LASTEXITCODE -ne 0) { throw "Kafka storage formatting failed." }
        Write-Host "Formatted Kafka KRaft storage with cluster ID $clusterId"
    }
    "start" {
        Write-Host "Starting Kafka in this terminal. Leave it open; use another PowerShell window for the application."
        $logConfigUri = ([System.Uri](Join-Path $KafkaHome "config\log4j2.yaml")).AbsoluteUri
        & java -Xms1G -Xmx1G "-Dlog4j2.configurationFile=$logConfigUri" -cp $javaClasspath kafka.Kafka $runtimeConfig
    }
    "create-topic" {
        & java -cp $javaClasspath org.apache.kafka.tools.TopicCommand --bootstrap-server $BootstrapServers --create --if-not-exists --topic $Topic --partitions 3 --replication-factor 1
        if ($LASTEXITCODE -ne 0) { throw "Kafka topic creation failed." }
    }
}
