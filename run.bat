@echo off
REM Launch Inflect Studio (Windows).
setlocal
cd /d "%~dp0"

set "VENV=.venv"

if not exist "%VENV%\Scripts\python.exe" (
    echo Creating virtualenv in %VENV% ...
    py -3 -m venv "%VENV%" || python -m venv "%VENV%"
    call "%VENV%\Scripts\activate.bat"
    echo.
    echo For GPU, install a CUDA build of torch FIRST, e.g.:
    echo     pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
    echo.
    python -m pip install --upgrade pip
    pip install -r requirements.txt
) else (
    call "%VENV%\Scripts\activate.bat"
)

python -m inflect %*
endlocal
