@echo off
setlocal

echo ======================================================================
echo STUBBLEAI - DAILY V1 PIPELINE
echo %date% %time%
echo ======================================================================

cd /d "%~dp0..\.."

set "PYTHON=%~dp0..\..\.venv\Scripts\python.exe"

echo.
echo [1/6] Updating FIRMS history...
"%PYTHON%" "scripts\v1\update_live_firms_2026.py"
if %errorlevel% neq 0 goto error

echo.
echo [2/6] Building district fire counts...
"%PYTHON%" "scripts\v1\live_district_fire_2026.py"
if %errorlevel% neq 0 goto error

echo.
echo [3/6] Building live fire features...
"%PYTHON%" "scripts\v1\build_live_2026_features.py"
if %errorlevel% neq 0 goto error

echo.
echo [4/6] Fetching weather forecast...
"%PYTHON%" "scripts\v1\live_weather_2026.py"
if %errorlevel% neq 0 goto error

echo.
echo [5/6] Running prediction...
"%PYTHON%" "scripts\v1\predict_2026.py"
if %errorlevel% neq 0 goto error

echo.
echo [6/6] Updating prediction tracker...
"%PYTHON%" "scripts\v1\prediction_tracker.py"
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
