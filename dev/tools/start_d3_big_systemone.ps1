<#
.SYNOPSIS
Run D3 Q4 System One on the RTX 3090, with its projector on the CPU.
.DESCRIPTION
Only the RTX 3090 is visible to this process. Automatic fitting can move
model layers to RAM. The explicit context size stays fixed during fitting.
ContextSize sets the limit per slot. Batch sizes cover one complete decision.
One slot avoids GPU allocation failures observed with two 8K slots.
The caller owns shutdown. A server failure exits this script without retry.
.PARAMETER ContextSize
Total tokens per decision, including state and question. Both batch sizes
default to this value. The server does not reduce this explicit limit.
.PARAMETER CpuTokenTable
Keep D3's input token table in RAM. This saves approximately 682 MiB of VRAM.
The separate embedding model is outside this script's scope.
.PARAMETER GpuReserveMiB
Free GPU memory target during startup fitting, in MiB. Later GPU consumers
can use that space. Restarting the server repeats fitting against free memory.
.PARAMETER CacheType
Type of both K and V cache. FP16 improved measured prefill in these tests.
The tested 16K RAM profile uses q8_0 with CpuTokenTable.
#>
param(
    [string]$Exe = "C:\workspace\llama13\llama-server.exe",
    [int]$Port = 8082,
    [string]$ModelDirectory = "C:\source\ai\d3",
    [int]$ContextSize = 8192,
    [int]$BatchSize = 0,
    [int]$MicroBatchSize = 0,
    [int]$Threads = 16,
    [int]$GpuReserveMiB = 1536,
    [string]$GpuLayers = "auto",
    [ValidateSet("f16", "q8_0")][string]$CacheType = "f16",
    [switch]$CpuTokenTable,
    [string]$LogFile = ""
)

$ErrorActionPreference = "Stop"
$model = Join-Path $ModelDirectory "vllm-d3-Q4_K_M.gguf"
$projector = Join-Path $ModelDirectory "mmproj-vllm-d3-BF16.gguf"
foreach ($path in @($Exe, $model, $projector)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Required file not found: $path"
    }
}
# D3 needs one physical batch for the full input, including question tokens.
if ($BatchSize -eq 0) { $BatchSize = $ContextSize }
if ($MicroBatchSize -eq 0) { $MicroBatchSize = $ContextSize }
if ($ContextSize -lt 4096 -or $MicroBatchSize -lt $ContextSize -or $BatchSize -lt $MicroBatchSize) {
    throw "ContextSize must be at least 4096. MicroBatchSize must cover ContextSize. BatchSize must cover MicroBatchSize."
}

$gpuRows = @(& nvidia-smi --query-gpu=name,uuid --format=csv,noheader)
if ($LASTEXITCODE -ne 0) { throw "Cannot inspect GPU identity." }
$selected = @($gpuRows | Where-Object { $_ -match '^NVIDIA GeForce RTX 3090\s*,' })
if ($selected.Count -ne 1) { throw "Expected exactly one RTX 3090." }
$gpuUuid = ($selected[0] -split ',')[1].Trim()

if (-not $LogFile) {
    $logDir = Join-Path $ModelDirectory "logs"
    New-Item -ItemType Directory -Force -Path $logDir | Out-Null
    $LogFile = Join-Path $logDir "d3-big_$(Get-Date -Format 'yyyyMMdd-HHmmss-fff').log"
}
$logParent = Split-Path -Parent ([IO.Path]::GetFullPath($LogFile))
New-Item -ItemType Directory -Force -Path $logParent | Out-Null

$serverArgs = @(
    "--host", "127.0.0.1", "--port", "$Port", "--alias", "d3-big",
    "--model", $model, "--mmproj", $projector,
    "--no-mmproj-offload", "--mmproj-device", "none",
    "--device", "CUDA0", "--split-mode", "none", "--main-gpu", "0",
    "--n-gpu-layers", $GpuLayers, "--fit", "on", "--fit-target", "$GpuReserveMiB",
    "--ctx-size", "$ContextSize", "--parallel", "1",
    "--batch-size", "$BatchSize", "--ubatch-size", "$MicroBatchSize",
    "--threads", "$Threads", "--threads-batch", "$Threads",
    "--flash-attn", "on", "--cache-type-k", $CacheType, "--cache-type-v", $CacheType,
    "--log-file", $LogFile, "--verbosity", "4"
)
if (-not $CpuTokenTable) {
    $serverArgs += @("--override-tensor", "token_embd.weight=CUDA0")
}

# The UUID mask prevents CUDA from placing tensors on the RTX 5090.
# CUDA0 denotes the RTX 3090 inside this process, regardless of system order.
$priorVisible = $env:CUDA_VISIBLE_DEVICES
$priorGraph = $env:GGML_CUDA_GRAPH_OPT
try {
    $env:CUDA_VISIBLE_DEVICES = $gpuUuid
    $env:GGML_CUDA_GRAPH_OPT = "0"
    Write-Host "Starting D3 System One on RTX 3090 ($gpuUuid). Projector: CPU."
    Write-Host "API: http://127.0.0.1:$Port/v1/systemone"
    Write-Host "Context: $ContextSize; slots: 1; batch: $BatchSize; microbatch: $MicroBatchSize"
    Write-Host "Log: $LogFile"
    & $Exe @serverArgs
    $serverExit = $LASTEXITCODE
    if ($serverExit -ne 0) { throw "D3 server exited with code $serverExit. Inspect $LogFile." }
}
finally {
    $env:CUDA_VISIBLE_DEVICES = $priorVisible
    $env:GGML_CUDA_GRAPH_OPT = $priorGraph
}
