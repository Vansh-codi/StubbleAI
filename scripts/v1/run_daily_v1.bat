@echo off
setlocal

echo ======================================================================
echo STUBBLEAI - DAILY V1 PIPELINE
echo %date% %time%
echo ======================================================================

cd /d "C:\Users\Vansh\Desktop\ai project"

set "PYTHON=C:\Users\Vansh\Desktop\ai project\.venv\Scripts\python.exe"

echo.
echo [1/6] Updating FIRMS history...
"%PYTHON%" update_live_firms_2026.py
if %errorlevel% neq 0 goto error

echo.
echo [2/6] Building district fire counts...
"%PYTHON%" live_district_fire_2026.py
if %errorlevel% neq 0 goto error

echo.
echo [3/6] Building live fire features...
"%PYTHON%" build_live_2026_features.py
if %errorlevel% neq 0 goto error

echo.
echo [4/6] Fetching weather forecast...
"%PYTHON%" live_weather_2026.py
if %errorlevel% neq 0 goto error

echo.
echo [5/6] Running prediction...
"%PYTHON%" predict_2026.py
if %errorlevel% neq 0 goto error

echo.
echo [6/6] Updating prediction tracker...
"%PYTHON%" prediction_tracker.py
if %errorlevel% neq 0 goto error

echo.
echo ======================================================================
echo PIPELINE COMPLETE - %date% %time%
echo ======================================================================
goto end

:error
echo.
echo ======================================================================
echo PIPELINE FAILED - %date% %time%
echo ======================================================================
exit /b 1

:end
exit /b 0
