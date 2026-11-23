param(
    [string]$InputPath = "data\processed\readings",
    [string]$OutputPath = "outputs\graphx",
    [double]$SimilarityThreshold = 0.98,
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$jarPath = Join-Path $root "scala\graphx-client-similarity\target\powerpulse-graphx-client-similarity.jar"
if (-not (Test-Path -LiteralPath $jarPath)) { throw "Build the GraphX module first with scripts\build_graphx.ps1." }
$overwriteText = if ($Overwrite) { "true" } else { "false" }
& (Join-Path $env:SPARK_HOME "bin\spark-submit.cmd") --class org.powerpulse.graph.ClientSimilarityGraph --master local[*] $jarPath (Join-Path $root $InputPath) (Join-Path $root $OutputPath) $SimilarityThreshold $overwriteText
if ($LASTEXITCODE -ne 0) { throw "GraphX execution failed with code $LASTEXITCODE." }
