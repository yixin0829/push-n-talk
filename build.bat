@echo off
echo Building PushToTalk GUI Windows Executable...
echo.

REM Clean previous .exe and .zip files
if exist "dist\PushToTalk.exe" del /f /q "dist\PushToTalk.exe"
if exist "dist\PushToTalk.zip" del /f /q "dist\PushToTalk.zip"

REM Build the executable
echo Building GUI application with PyInstaller...
uv run pyinstaller push_to_talk.spec

REM Check if build was successful
if exist "dist\PushToTalk.exe" (
    echo.
    echo ========================================
    echo Build successful!
    echo GUI Executable created: dist\PushToTalk.exe
    echo ========================================
    echo.
    REM Compress the executable to a zip file for easy distribution
    echo Compressing executable to PushToTalk.zip...
    powershell -Command "Compress-Archive -Path dist\PushToTalk.exe -DestinationPath dist\PushToTalk.zip"
    echo Executable compressed to PushToTalk.zip
    echo.

    echo To run the GUI application:
    echo 1. Double-click dist\PushToTalk.exe
    echo 2. Configure your OpenAI API key in the setup window
    echo 3. Adjust audio and hotkey settings as needed
    echo 4. Click "Start Application" to begin
    echo 5. Run as Administrator for hotkey detection
    echo.
    echo The application will show configuration dialogs on first run
    echo and then run in the background with your configured hotkeys.
    echo.

) else (
    echo.
    echo ========================================
    echo Build failed! Check the output above for errors.
    echo ========================================
    echo.
    echo Common issues:
    echo - Make sure uv is installed and working
    echo - Check that all dependencies are available
    echo - Ensure PyInstaller is in the dev dependencies
    echo.
)

pause
