# Run Static Baseline Analysis
# This script runs the complete workflow for the static baseline model:
# 1. Trains the model (if not already trained)
# 2. Evaluates the model
# 3. Visualizes the agent's behavior

$ErrorActionPreference = "Stop"

# Resolve and switch to the project root so relative paths always work
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

# Use project-local virtual environment Python explicitly
$PythonExe = Join-Path $ScriptDir ".venv\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    throw "Python executable not found at $PythonExe. Recreate venv first."
}

# Set PYTHONPATH to include the project root
$env:PYTHONPATH = $ScriptDir

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "       PPO Maze Navigation Analysis         " -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan

# 1. Check if model exists, if not, train it
$ModelPath = "models/static_baseline/seed_42/best_model.zip"

if (-not (Test-Path $ModelPath)) {
    Write-Host "`n[1/3] Training Static Baseline Model..." -ForegroundColor Yellow
    & $PythonExe src/training/train_static.py --timesteps 500000 --seed 42 --profile tuned --env-max-steps 400 --mistake-log logs/mistakes/static_eval_mistakes.jsonl
} else {
    Write-Host "`n[1/3] Model already exists at $ModelPath. Skipping training." -ForegroundColor Green
}

# 2. Evaluate Model
Write-Host "`n[2/3] Evaluating Model..." -ForegroundColor Yellow
& $PythonExe src/evaluation/evaluate_model.py --model models/static_baseline/seed_42/best_model --episodes 20 --mistake-log logs/mistakes/static_eval_mistakes.jsonl

# 2.5 Analyze mistakes before visualization
Write-Host "`n[2.5/3] Analyzing Mistakes..." -ForegroundColor Yellow
& $PythonExe src/analysis/failure_analysis.py --log logs/mistakes/static_eval_mistakes.jsonl

# 3. Visualize Trajectory
Write-Host "`n[3/3] Visualizing Trajectory..." -ForegroundColor Yellow
& $PythonExe src/visualization/visualize_maze.py --model models/static_baseline/seed_42/best_model --max-steps 400

Write-Host "`n============================================" -ForegroundColor Cyan
Write-Host "Analysis Complete!" -ForegroundColor Cyan
Write-Host "View results in:"
Write-Host "  - Evaluation: Console output above"
Write-Host "  - Visualization: figures/trajectories/static_example.png"
Write-Host "============================================" -ForegroundColor Cyan
