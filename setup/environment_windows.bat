@echo off

:: Create a virtual environment
python -m venv env

:: Activate the virtual environment
call env\Scripts\activate.bat

:: Upgrade pip
pip install --upgrade pip

:: Install the required packages
pip install -r setup\requirements.txt

:: Ask the user if they want to install demo requirements
set /p install_demo="Do you also want to install requirements requred to run demo? (y/n): "

if /i "%install_demo%"=="y" (
    pip install -r setup\demo_requirements.txt
    echo Demo requirements installed.
) else (
    echo Skipping demo requirements installation.
)

echo Environment setup complete. To activate the virtual environment, run "env\Scripts\activate".
