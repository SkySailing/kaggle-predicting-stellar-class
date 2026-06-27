#!/bin/bash
# Installation script for improved pipeline

echo "=========================================="
echo "Installing required packages..."
echo "=========================================="

# Install required packages
pip install numpy pandas scikit-learn matplotlib seaborn

echo ""
echo "Installing XGBoost..."
pip install xgboost

echo ""
echo "Installing LightGBM..."
pip install lightgbm

echo ""
echo "Installing CatBoost..."
pip install catboost

echo ""
echo "=========================================="
echo "Installation complete!"
echo "=========================================="
echo ""
echo "To run the improved pipeline:"
echo "  python3 run_improved_pipeline.py"
echo ""
echo "Expected runtime: 20-25 minutes"
echo "Expected score improvement: +0.002-0.003"
echo "Target score: 0.9725+"
