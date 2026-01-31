@echo off
REM Start Polymarket data collector for XRP trades only (Windows)
REM Run this script to start continuous XRP data collection

echo Starting Polymarket XRP Data Collector...
echo Filtering for XRP-related trades only
echo Press Ctrl+C to stop

cd /d "%~dp0\.."
python src/data/collector.py start --interval 300 --market xrp

pause
