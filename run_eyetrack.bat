@echo off
rem First try the default 'python' in the system PATH.
rem The script itself will automatically locate and redirect to a compatible
rem Python version (like Python 3.11) if the default one is missing dependencies.
python "%~dp0eyetrack\eye_3d_rgbd.py" %*
if %errorlevel% neq 0 (
    echo.
    echo Default 'python' failed. Trying fallback to explicit Python 3.11 path...
    "C:\Users\Syncard\AppData\Local\Programs\Python\Python311\python.exe" "%~dp0eyetrack\eye_3d_rgbd.py" %*
)
pause
