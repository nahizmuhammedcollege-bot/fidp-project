@echo off
title Automatic Number Plate Recognition (ANPR) System
color 0A
cls
echo ===============================================================================
echo            AUTOMATIC NUMBER PLATE RECOGNITION (ANPR) SYSTEM
echo              Image Processing & Computer Vision Final Project
echo ===============================================================================
echo.
echo Launching ANPR Pipeline and Benchmark Evaluation on Kaggle Dataset...
echo.

python main.py

echo.
echo ===============================================================================
echo Execution finished! Opening results directory...
echo ===============================================================================
start "" "%~dp0results"
pause
