param(
  [Parameter(Mandatory=$false)][string]$CertPath = ".\caddy_root.crt",
  [Parameter(Mandatory=$false)][switch]$LocalMachine
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $CertPath)) {
  Write-Host "Certificate not found at: $CertPath" -ForegroundColor Red
  Write-Host "Tip: docker cp fullbuild-gateway-1:/data/caddy/pki/authorities/local/root.crt .\caddy_root.crt"
  exit 1
}

$store = if ($LocalMachine) { "Cert:\LocalMachine\Root" } else { "Cert:\CurrentUser\Root" }

try {
  $result = Import-Certificate -FilePath $CertPath -CertStoreLocation $store
  Write-Host "Imported Caddy root certificate into $store" -ForegroundColor Green
  Write-Host "Thumbprint: $($result.Thumbprint)"
  if (-not $LocalMachine) {
    Write-Host "If your browser still warns, close & reopen the browser." -ForegroundColor Yellow
  } else {
    Write-Host "LocalMachine install requires running PowerShell as Administrator." -ForegroundColor Yellow
  }
} catch {
  Write-Host "Failed to import certificate into $store" -ForegroundColor Red
  Write-Host $_.Exception.Message -ForegroundColor Red
  if (-not $LocalMachine) {
    Write-Host "Try running: Import-Certificate -FilePath $CertPath -CertStoreLocation Cert:\CurrentUser\Root" -ForegroundColor Yellow
  } else {
    Write-Host "Try running an elevated (Admin) PowerShell, or omit -LocalMachine to install for CurrentUser." -ForegroundColor Yellow
  }
  exit 1
}
