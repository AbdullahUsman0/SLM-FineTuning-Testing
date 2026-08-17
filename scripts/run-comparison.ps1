param(
    [string]$Label = "local-qwen-0.8b-q4",
    [string]$CaseFile = "evaluation\comparison-v1.jsonl",
    [string]$BaseUrl = "http://127.0.0.1:8080/v1",
    [int]$MaxTokens = 256
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$casePath = Join-Path $projectRoot $CaseFile
$resultsDirectory = Join-Path $projectRoot "results"
$systemPrompt = "You are a forecasting requirements assistant. Be concise, do not invent missing values, and ask at most one clarification question at a time."

try {
    $health = Invoke-RestMethod -Uri ($BaseUrl.Replace("/v1", "/health")) -TimeoutSec 5
    if ($health.status -ne "ok") {
        throw "Server status is '$($health.status)'"
    }
}
catch {
    throw "The local model server is not ready. Start it in another PowerShell window with .\scripts\start-local-server.ps1"
}

if (-not (Test-Path -LiteralPath $casePath)) {
    throw "Test-case file is missing: $casePath"
}

$testCases = Get-Content -LiteralPath $casePath -Encoding UTF8 |
    Where-Object { $_.Trim() } |
    ForEach-Object { $_ | ConvertFrom-Json }

$results = foreach ($testCase in $testCases) {
    Write-Host "Running $($testCase.id)..."
    $timer = [System.Diagnostics.Stopwatch]::StartNew()
    $responseText = $null
    $errorText = $null
    $checks = [ordered]@{}

    try {
        $body = @{
            model = "local-qwen"
            messages = @(
                @{ role = "system"; content = $systemPrompt }
                @{ role = "user"; content = $testCase.prompt }
            )
            temperature = 0.0
            max_tokens = $MaxTokens
            stream = $false
        } | ConvertTo-Json -Depth 8

        $response = Invoke-RestMethod `
            -Method Post `
            -Uri "$($BaseUrl.TrimEnd('/'))/chat/completions" `
            -ContentType "application/json" `
            -Body $body `
            -TimeoutSec 120
        $responseText = [string]$response.choices[0].message.content
        $normalized = $responseText.ToLowerInvariant()

        $containsRequired = $true
        if ($testCase.PSObject.Properties.Name -contains "must_contain") {
            foreach ($required in @($testCase.must_contain)) {
                if (-not $normalized.Contains(([string]$required).ToLowerInvariant())) {
                    $containsRequired = $false
                }
            }
        }
        $checks["contains_required_text"] = $containsRequired

        if ($testCase.PSObject.Properties.Name -contains "must_contain_any") {
            $containsAny = $false
            foreach ($option in @($testCase.must_contain_any)) {
                if ($normalized.Contains(([string]$option).ToLowerInvariant())) {
                    $containsAny = $true
                }
            }
            $checks["contains_any_expected_text"] = $containsAny
        }

        $avoidsForbidden = $true
        if ($testCase.PSObject.Properties.Name -contains "must_not_contain") {
            foreach ($forbidden in @($testCase.must_not_contain)) {
                if ($normalized.Contains(([string]$forbidden).ToLowerInvariant())) {
                    $avoidsForbidden = $false
                }
            }
        }
        $checks["avoids_forbidden_text"] = $avoidsForbidden

        if ($testCase.PSObject.Properties.Name -contains "max_question_marks") {
            $questionCount = [regex]::Matches($responseText, "\?").Count
            $checks["question_limit"] = $questionCount -le [int]$testCase.max_question_marks
        }

        if ($testCase.PSObject.Properties.Name -contains "max_numbered_items") {
            $numberedItems = [regex]::Matches($responseText, "(?m)^\s*\d+[.)]").Count
            $checks["numbered_item_limit"] = $numberedItems -le [int]$testCase.max_numbered_items
        }
    }
    catch {
        $errorText = $_.Exception.Message
        $checks["request_succeeded"] = $false
    }
    finally {
        $timer.Stop()
    }

    $passed = $true
    foreach ($value in $checks.Values) {
        if (-not $value) { $passed = $false }
    }

    [pscustomobject]@{
        id = $testCase.id
        category = $testCase.category
        prompt = $testCase.prompt
        expected_behavior = $testCase.expected_behavior
        automated_pass = $passed
        checks = [pscustomobject]$checks
        latency_ms = [math]::Round($timer.Elapsed.TotalMilliseconds, 1)
        response = $responseText
        error = $errorText
        human_score_0_to_2 = $null
        human_notes = $null
    }
}

$timestamp = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssZ")
$safeLabel = $Label -replace "[^A-Za-z0-9._-]", "-"
$baseName = "$safeLabel-comparison-v1-$timestamp"
$jsonPath = Join-Path $resultsDirectory "$baseName.json"
$csvPath = Join-Path $resultsDirectory "$baseName.csv"
$passCount = @($results | Where-Object automated_pass).Count

$report = [ordered]@{
    schema_version = 1
    suite = "comparison-v1"
    created_at_utc = $timestamp
    system_label = $Label
    model = "local-qwen"
    temperature = 0.0
    max_tokens = $MaxTokens
    case_count = @($results).Count
    automated_passes = $passCount
    results = @($results)
}

$report | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding UTF8
$results |
    Select-Object id, category, prompt, expected_behavior, automated_pass, latency_ms, response, error, human_score_0_to_2, human_notes |
    Export-Csv -LiteralPath $csvPath -NoTypeInformation -Encoding UTF8

Write-Host ""
Write-Host "Automated result: $passCount/$(@($results).Count)"
Write-Host "JSON: $jsonPath"
Write-Host "Review CSV: $csvPath"
Write-Host "Open the CSV and fill human_score_0_to_2 and human_notes before comparing systems."
