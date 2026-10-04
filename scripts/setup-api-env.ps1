<#
.SYNOPSIS
    Sets up a PowerShell session for the API experiments (exp3_compare.py).

.DESCRIPTION
    - checks Python (3.10+) and creates/activates a venv. No packages are needed:
      the API path uses only the standard library, so torch/transformers from
      participant/requirements.txt are NOT installed here.
    - asks for the API key without echoing it and keeps it in this session only
      (never written to disk).
    - clears EVALUATION_* variables, which would override the config files.
    - checks that the repo files, the val data and the organiser package exist.
    - sends one tiny request to check that the key and the model work.
    - prints the commands to run.

.USAGE
    Run it from the repo root, dot-sourced so the settings stay in your window:

        Set-ExecutionPolicy -Scope Process Bypass
        . .\scripts\setup-api-env.ps1 -Participant "C:\path\to\participant"

    Options:
        -Provider hf | siliconflow    (default hf = Hugging Face router)
        -Model "Qwen/Qwen3-8B:nscale" (default Qwen/Qwen3-8B:nscale for hf)
        -SkipCheck                     skip the test request
#>

param(
    [Parameter(Mandatory = $true)]
    [string]$Participant,

    [ValidateSet("hf", "siliconflow")]
    [string]$Provider = "hf",

    [string]$Model = "",

    [string]$Repo = (Get-Location).Path,

    [string]$Venv = "$HOME\venvs\migu",

    [switch]$SkipCheck
)

$ErrorActionPreference = "Stop"

function Write-Ok($msg)   { Write-Host "  [ok]   $msg" -ForegroundColor Green }
function Write-Warn($msg) { Write-Host "  [warn] $msg" -ForegroundColor Yellow }
function Write-Fail($msg) { Write-Host "  [fail] $msg" -ForegroundColor Red }

# ---- provider presets ---------------------------------------------------------
$presets = @{
    "hf" = @{
        BaseUrl = "https://router.huggingface.co/v1"
        KeyEnv  = "HF_TOKEN"
        Model   = "Qwen/Qwen3-8B:nscale"
        Flags   = "--soft-switch --key-env HF_TOKEN"
        KeyHint = "hf_..."
    }
    "siliconflow" = @{
        BaseUrl = "https://api.siliconflow.com/v1"
        KeyEnv  = "SILICONFLOW_API_KEY"
        Model   = "Qwen/Qwen3-8B"
        Flags   = "--stream"
        KeyHint = "sk-..."
    }
}
$p = $presets[$Provider]

# fail early on a placeholder or wrong participant path
$badChars = [IO.Path]::GetInvalidPathChars() + [char[]]'<>|"*?'
if ($Participant.IndexOfAny($badChars) -ge 0 -or
    -not (Test-Path -LiteralPath (Join-Path $Participant "run_inference.py"))) {
    Write-Host "`n  [fail] -Participant must be the real folder that contains run_inference.py" -ForegroundColor Red
    Write-Host "         got: $Participant" -ForegroundColor Red
    Write-Host "         find it with:" -ForegroundColor Red
    Write-Host "         Get-ChildItem `$HOME -Recurse -Filter run_inference.py -ErrorAction SilentlyContinue | Select-Object FullName"
    Write-Host "         nothing found = you don't have the organiser package yet; ask Ismail F. for it`n"
    return
}
if (-not $Model) { $Model = $p.Model }

Write-Host "`nAPI experiment setup ($Provider, $Model)`n" -ForegroundColor Cyan

# ---- 1. Python + venv ---------------------------------------------------------
Write-Host "1. Python"
$python = $null
foreach ($cand in @("py -3.12", "py -3", "python")) {
    try {
        $exe, $rest = $cand.Split(" ")
        $rest = @($rest | Where-Object { $_ })
        $ver  = & $exe @rest -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
        if ($LASTEXITCODE -eq 0 -and $ver) {
            $major, $minor = $ver.Trim().Split(".")
            if ([int]$major -eq 3 -and [int]$minor -ge 10) { $python = $cand; break }
        }
    } catch { }
}
if (-not $python) {
    Write-Fail "Python 3.10+ not found. Install Python 3.12 from python.org (tick 'Add to PATH'), then rerun."
    return
}
Write-Ok "found Python $ver ($python)"

if (-not (Test-Path "$Venv\Scripts\Activate.ps1")) {
    Write-Host "  creating venv at $Venv ..."
    $exe, $rest = $python.Split(" ")
    $rest = @($rest | Where-Object { $_ })
    & $exe @rest -m venv $Venv
    if ($LASTEXITCODE -ne 0) { Write-Fail "venv creation failed"; return }
}
. "$Venv\Scripts\Activate.ps1"
Write-Ok "venv active: $Venv"

# UTF-8 so Chinese text doesn't crash the Windows console
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

# ---- 2. API key ---------------------------------------------------------------
Write-Host "`n2. API key"
Get-ChildItem Env: | Where-Object { $_.Name -like "EVALUATION_*" } |
    ForEach-Object { Remove-Item "Env:$($_.Name)"; Write-Warn "cleared $($_.Name)" }

$existing = [Environment]::GetEnvironmentVariable($p.KeyEnv, "Process")
if ($existing) {
    Write-Ok "$($p.KeyEnv) already set in this session"
} else {
    $secure = Read-Host "  Paste your $Provider key ($($p.KeyHint)); it won't be shown" -AsSecureString
    $bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
    try { $plain = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr).Trim() }
    finally { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr) }
    if (-not $plain) { Write-Fail "no key entered"; return }
    Set-Item -Path "Env:$($p.KeyEnv)" -Value $plain
    Write-Ok "$($p.KeyEnv) set for this window only (starts with $($plain.Substring(0, [Math]::Min(5, $plain.Length)))..., $($plain.Length) chars)"
}

# ---- 3. files -----------------------------------------------------------------
Write-Host "`n3. Files"
$missing = 0
$checks = [ordered]@{
    "$Participant\run_inference.py"            = "organiser package (ask Ismail F. for the participant folder)"
    "$Repo\scripts\exp3_compare.py"            = "repo script (git pull on main)"
    "$Repo\scripts\validate.py"                = "Yahya's validator (git pull on main)"
    "$Repo\scripts\eval.py"                    = "Yahya's scorer (git pull on main)"
    "$Repo\data\val_input_n50.jsonl"           = "val inputs (data/ is gitignored: get it from Ismail F. or run Zakaria's convert_val.py --n 50)"
    "$Repo\data\val_gold_n50.jsonl"            = "val gold (same as above)"
}
foreach ($f in $checks.Keys) {
    if (Test-Path -LiteralPath $f) { Write-Ok (Split-Path $f -Leaf) }
    else { Write-Fail "missing $f  ->  $($checks[$f])"; $missing++ }
}

# ---- 4. test request ----------------------------------------------------------
if (-not $SkipCheck) {
    Write-Host "`n4. Test request (a few tokens, costs almost nothing)"
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    $key  = [Environment]::GetEnvironmentVariable($p.KeyEnv, "Process")
    $body = @{
        model      = $Model
        messages   = @(@{ role = "user"; content = "Reply with the single word OK. /no_think" })
        max_tokens = 20
    } | ConvertTo-Json -Depth 5
    try {
        $t0 = Get-Date
        $r = Invoke-RestMethod -Method Post -Uri "$($p.BaseUrl)/chat/completions" `
            -Headers @{ Authorization = "Bearer $key" } -ContentType "application/json" -Body $body -TimeoutSec 60
        $secs = [Math]::Round(((Get-Date) - $t0).TotalSeconds, 1)
        Write-Ok "model answered in $secs s: '$($r.choices[0].message.content.Trim())'"
    } catch {
        $err  = $_
        $code = $null
        if ($err.Exception.Response) { $code = [int]$err.Exception.Response.StatusCode }
        switch ($code) {
            401     { Write-Fail "401: key is wrong or expired. Rerun after: Remove-Item Env:$($p.KeyEnv)" }
            402     { Write-Fail "402: no credit left on this account" }
            403     { Write-Fail "403: blocked (network/Cloudflare) or the key lacks 'inference providers' permission" }
            400     { Write-Fail "400: model name not accepted: $Model (check the providers on the model page)" }
            default { Write-Fail "request failed: $($err.Exception.Message)" }
        }
        $missing++
    }
}

# ---- 5. what to run -----------------------------------------------------------
$flags = $p.Flags
$baseArgs = "--participant `"$Participant`" ``
      --input data\val_input_n50.jsonl --gold data\val_gold_n50.jsonl ``
      --scripts scripts --base-url $($p.BaseUrl) $flags"

Write-Host "`n5. Commands" -ForegroundColor Cyan
if ($missing -gt 0) { Write-Warn "fix the $missing problem(s) above first" }
Write-Host @"

  # non-thinking run on the 50 val points
  python scripts\exp3_compare.py $baseArgs ``
      --out experiments\<expNNN-name> --modes nothink --model "$Model"

  # results: experiments\<expNNN-name>\comparison.md and out_nothink\metrics.json
  # never paste the key in chat or commit it; it disappears when you close this window
"@
