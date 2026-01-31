@echo off
REM Start Polymarket data collector (Windows)
REM Run this script to start continuous data collection

echo Starting Polymarket Data Collector...
echo Press Ctrl+C to stop

cd /d "%~dp0\.."
python src/data/collector.py start --interval 300

pause
