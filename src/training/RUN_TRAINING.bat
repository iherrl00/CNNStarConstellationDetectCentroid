@echo off
title Fine-Tuning MobileUNet 
echo ==========================================================
echo Iniciando Fine Tuning
echo ==========================================================
echo.
echo [Hardware] Optimizando para +8gb VRAM
echo [Dataloader] num_workers=8, pin_memory=True
echo [Dataset] Leyendo directamente desde src\data_generation\training_data
echo [Estrategia] Learning Rate hiper-bajo (1e-5) para retener pesos base.
echo.

cd /d "%~dp0"

:: Ejecutando con calculo de metricas reales del dataset v2 y LR bajo
..\..\.venv\Scripts\python.exe training_stepLR.py ^
    --trial 2 ^
    --ep 40 ^
    --batch_size 20 ^
    --create 0 ^
    --load "saved_models\MobileUNet_2_40.pt" ^
    --lr 1e-5 ^
    --compute_stats 1 ^
    --pin_memory 1

echo.
echo Entrenamiento finalizado.
pause
