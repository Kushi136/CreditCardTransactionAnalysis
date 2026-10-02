@echo off
title Credit Card Fraud Detection Web Server
echo ========================================================
echo Starting Credit Card Transaction Analytics Web Dashboard
echo ========================================================
echo.
cd /d "%~dp0"
python app\app.py
pause
