@echo off
echo Running Experiment 4 Pipeline...
python scripts\pipeline.py

if errorlevel 1 (
    echo Pipeline failed.
    exit /b 1
)

echo Pipeline executed successfully.
pause