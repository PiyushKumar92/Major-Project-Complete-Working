@echo off
echo ========================================
echo Restarting Flask Application
echo ========================================
echo.

echo Stopping any running Flask processes...
taskkill /F /IM python.exe /FI "WINDOWTITLE eq *run_app.py*" 2>nul
timeout /t 2 /nobreak >nul

echo.
echo Starting Flask Application...
echo.
start "Missing Person System" cmd /k "cd /d "%~dp0" && python run_app.py"

echo.
echo ========================================
echo Server is starting...
echo Wait 10 seconds then access:
echo http://localhost:5000/features
echo ========================================
echo.
pause
