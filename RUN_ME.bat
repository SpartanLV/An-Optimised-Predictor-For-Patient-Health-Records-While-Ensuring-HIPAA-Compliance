@echo off
setlocal
cd /d "%~dp0"

echo ============================================
echo CSE400 Full Setup + Cert + Synthea Ingest
echo ============================================
echo.

REM Try Windows PowerShell (full path)
set "PS1=%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe"
if exist "%PS1%" (
  "%PS1%" -NoProfile -ExecutionPolicy Bypass -File "%~dp0RUN_ME.ps1"
  goto :done
)

REM Fallback: PowerShell 7 (pwsh) if installed
where pwsh >nul 2>&1
if %errorlevel%==0 (
  pwsh -NoProfile -ExecutionPolicy Bypass -File "%~dp0RUN_ME.ps1"
  goto :done
)

echo ERROR: PowerShell not found.
echo - Expected: %SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe
echo - Or PowerShell 7: pwsh
echo.
echo Install/enable PowerShell and try again.
echo.

:done
echo.
echo ============================================
echo Finished (see messages above).
echo ============================================
pause
endlocal
