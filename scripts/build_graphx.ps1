$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$sparkJars = Join-Path $env:SPARK_HOME "jars"
$source = Join-Path $root "scala\graphx-client-similarity\src\main\scala\org\powerpulse\graph\ClientSimilarityGraph.scala"
$target = Join-Path $root "scala\graphx-client-similarity\target"
$classes = Join-Path $target "classes"
$jarPath = Join-Path $target "powerpulse-graphx-client-similarity.jar"

if (-not (Test-Path -LiteralPath $sparkJars)) { throw "SPARK_HOME jars were not found: $sparkJars" }
if (-not (Test-Path -LiteralPath $source)) { throw "GraphX source was not found: $source" }
New-Item -ItemType Directory -Force -Path $classes | Out-Null
New-Item -ItemType Directory -Force -Path $target | Out-Null

# Java expands the quoted jars\* classpath. -usejavacp makes that classpath
# available to scalac without expanding hundreds of jars in the Windows shell.
& java -cp (Join-Path $sparkJars "*") scala.tools.nsc.Main -usejavacp -d $classes $source
if ($LASTEXITCODE -ne 0) { throw "Scala GraphX compilation failed." }
if (Test-Path -LiteralPath $jarPath) { Remove-Item -LiteralPath $jarPath -Force }
& jar --create --file $jarPath -C $classes .
if ($LASTEXITCODE -ne 0) { throw "GraphX jar packaging failed." }
Write-Host "Built $jarPath"
