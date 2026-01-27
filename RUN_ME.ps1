# Double-click runner (called by RUN_ME.bat)
$ErrorActionPreference = "Stop"

function Say($m){ Write-Host $m -ForegroundColor Cyan }

Set-Location -Path $PSScriptRoot

Say "==> docker compose up -d --build"
docker compose up -d --build

# Detect gateway container name reliably
Say "==> Detecting gateway container..."
$gateway = (docker ps --format "{{.Names}}" | Where-Object { $_ -match "gateway" } | Select-Object -First 1)
if (-not $gateway) { throw "Gateway container not found. Is docker compose running?" }
Say "Gateway container: $gateway"

# Copy Caddy root cert from gateway container
Say "==> Copying Caddy root certificate..."
$crt = Join-Path $PSScriptRoot "caddy_root.crt"
docker cp "$gateway`:/data/caddy/pki/authorities/local/root.crt" "$crt" | Out-Null
Say "Saved: $crt"

# Install cert into CurrentUser Root (no admin needed)
Say "==> Installing certificate to CurrentUser Root store..."
$cert = New-Object System.Security.Cryptography.X509Certificates.X509Certificate2($crt)
$store = New-Object System.Security.Cryptography.X509Certificates.X509Store("Root","CurrentUser")
$store.Open([System.Security.Cryptography.X509Certificates.OpenFlags]::ReadWrite)
$store.Add($cert)
$store.Close()
Say "Certificate installed for current user."

# Run ingest (single line; avoids PowerShell backtick paste issues)
Say "==> Running Synthea ingest..."
docker compose exec tools python scripts/ingest_synthea_csv.py --csv-dir /repo/data/synthea/csv --base-url http://patient-store:8002 --auth-url http://auth:8004 --username admin --password admin123 --limit-patients 500

Say "✅ All done."
Say "Open: https://localhost:8443/"
