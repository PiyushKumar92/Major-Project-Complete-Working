@echo off
title FYMP Investigation Platform - Complete Startup
color 0A

echo ========================================
echo   FYMP Investigation Platform Startup
echo ========================================
echo.

echo [1/4] Starting Redis Server...
start "Redis Server" cmd /k "redis-server && echo Redis started successfully"
timeout /t 3 /nobreak >nul

echo [2/4] Starting Celery Worker...
start "Celery Worker" cmd /k "celery -A tasks.celery worker --loglevel=info --pool=solo"
timeout /t 5 /nobreak >nul

echo [3/4] Setting up environment...
call venv\Scripts\activate.bat 2>nul
set PYTHONPATH=%PYTHONPATH%;%CD%

echo [4/4] Starting Flask Application...
echo.
echo ========================================
echo   All services starting...
echo   Please wait for all windows to load
echo ========================================
echo.
echo Access Points:
echo   Main App: http://localhost:5000
echo   Admin Panel: http://localhost:5000/admin/dashboard
echo   Default Login: admin / admin123
echo.
echo Press any key to start Flask app...
pause >nul

python wsgi.py