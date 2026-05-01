@echo off
setlocal EnableExtensions

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start_dashboard.ps1" %*
exit /b %ERRORLEVEL%
