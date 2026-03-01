@echo off
:: Add OpenAgent directory to user PATH

set "AGENT_DIR=%~dp0"
:: Remove trailing backslash
set "AGENT_DIR=%AGENT_DIR:~0,-1%"

:: Check if already in PATH
echo %PATH% | findstr /I /C:"%AGENT_DIR%" >nul 2>&1
if %errorlevel%==0 (
    echo OpenAgent is already in PATH.
    goto :deps
)

:: Add to user PATH permanently
for /f "tokens=2*" %%A in ('reg query "HKCU\Environment" /v Path 2^>nul') do set "USER_PATH=%%B"
if defined USER_PATH (
    setx PATH "%USER_PATH%;%AGENT_DIR%"
) else (
    setx PATH "%AGENT_DIR%"
)
echo Added %AGENT_DIR% to PATH.
echo Restart your terminal for changes to take effect.

:deps
echo.
echo Installing dependencies...
pip install -r "%AGENT_DIR%\requirements.txt"
echo.
echo Done! You can now run: open-agent  or  oa
