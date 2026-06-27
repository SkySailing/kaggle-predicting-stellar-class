# SIMPLE SOLUTION - Manual Run Instructions

## The Problem
The improved pipeline is working but takes 20-25 minutes to complete, which is too long for automated testing.

## Solution: Run It Manually

### Option 1: Run the Python Script (Recommended)
```bash
# Clean up old processes first (if needed)
pkill -f "python3.*run_improved_pipeline"

# Run the pipeline
python3 run_improved_pipeline.py

# This will take 20-25 minutes and show progress like:
# [1/7] Loading data...
# [2/7] Creating advanced features...
# [3/7] Preparing data...
# [4/7] Training XGBoost...
#   Fold 1: 0.971234
#   ...
# [7/7] Creating ensemble...
# ENSEMBLE OOF Score: 0.972XXX
```

### Option 2: Quick Test with Fewer Folds (5 minutes)
If you want to test quickly, edit the script and change all instances of `n_splits=5` to `n_splits=2`:

```bash
# Make a quick test version
sed 's/n_splits=5/n_splits=2/g' run_improved_pipeline.py > run_quick_test.py
python3 run_quick_test.py
```

### Option 3: Use the Notebook on Kaggle
Upload the `stellar_class_improved.ipynb` to Kaggle and run it there with GPU:
1. Go to https://www.kaggle.com/code
2. Click "New Notebook"
3. Upload `stellar_class_improved.ipynb`
4. Change runtime to GPU
5. Run all cells

## Expected Output

When complete, you'll see:
```
======================================================================
ENSEMBLE OOF Score: 0.972XXX
======================================================================
Improvement over best single model: +0.000XXX

SUBMISSION CREATED: submission_improved.csv
======================================================================

Expected LB score: ~0.97XXX
Your current score: 0.97101
Expected improvement: +0.00XXX

To submit:
  kaggle competitions submit -c playground-series-s6e6 \
    -f submission_improved.csv \
    -m 'Ensemble: XGB+LGB+CAT with advanced features'
```

## After Running

Submit your improved result:
```bash
kaggle competitions submit -c playground-series-s6e6 \
  -f submission_improved.csv \
  -m "Ensemble: XGB+LGB+CAT with advanced features"
```

## If You Get Errors

### Import Error
```bash
pip install numpy pandas scikit-learn xgboost lightgbm catboost matplotlib seaborn
```

### Memory Error
Edit script and reduce folds to 3:
```python
skf = StratifiedKFold(n_splits=3, ...)  # instead of 5
```

### Path Error
The script now auto-detects these paths:
- `Data/playground-series-s6e6/` (your local path) ✅
- Current directory `.`/` (you also have data here) ✅
- `/kaggle/input/playground-series-s6e6/` (Kaggle notebooks)

## Bottom Line

**Just run this and wait 20-25 minutes:**
```bash
python3 run_improved_pipeline.py
```

The script will automatically:
- Find your data (in Data/playground-series-s6e6/ or current dir)
- Create 30+ features
- Train 3 models (XGB, LGB, CAT)
- Create ensemble
- Generate submission_improved.csv

**Expected score: 0.9725-0.9734** (vs your current 0.97101)
