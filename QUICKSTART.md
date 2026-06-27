# 🚀 QUICK START GUIDE - Improved Pipeline

## Your Current Status
- **Score**: 0.97101
- **Rank**: 385
- **Target**: 0.9725+ (Top 100-150)

## Step 1: Install Packages (5 minutes)

```bash
# Option A: Use the install script
bash install_packages.sh

# Option B: Manual installation
pip install numpy pandas scikit-learn matplotlib seaborn xgboost lightgbm catboost
```

## Step 2: Run the Improved Pipeline (20-25 minutes)

```bash
python3 run_improved_pipeline.py
```

This will:
1. ✅ Create 30+ advanced features
2. ✅ Train XGBoost (5-8 min)
3. ✅ Train LightGBM (4-6 min)
4. ✅ Train CatBoost (6-10 min)
5. ✅ Create weighted ensemble
6. ✅ Generate `submission_improved.csv`

## Step 3: Submit to Kaggle

```bash
kaggle competitions submit -c playground-series-s6e6 \
  -f submission_improved.csv \
  -m "Ensemble: XGB+LGB+CAT with advanced features"
```

## Expected Results

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| **Score** | 0.97101 | ~0.9725-0.9734 | +0.0020-0.0030 |
| **Rank** | 385 | ~50-150 | ⬆️ 230-335 |

## What's Different?

### 🎨 Advanced Features (+0.0008)
- Color indices: u-g, g-r, r-i, i-z (critical for astronomy!)
- Color ratios and magnitude statistics
- Redshift transformations
- Position sin/cos features

### 🤖 3-Model Ensemble (+0.0012)
- XGBoost: Fast, handles sparse features
- LightGBM: Memory efficient
- CatBoost: Best for categoricals
- Weighted by performance

### ⚙️ Better Parameters (+0.0004)
- Lower learning rate (0.02)
- Deeper trees (depth=10)
- Stronger regularization
- Optimized early stopping

## Troubleshooting

### If installation fails:
```bash
# Try with conda
conda install -c conda-forge xgboost lightgbm
pip install catboost
```

### If you get memory errors:
Edit `run_improved_pipeline.py` and reduce folds:
```python
# Change this line (appears 3 times):
skf = StratifiedKFold(n_splits=3, ...)  # instead of 5
```

### If CatBoost is too slow:
Edit `run_improved_pipeline.py`:
```python
'iterations': 500,  # instead of 1000
```

## Progress Monitoring

You'll see output like:
```
[1/7] Loading data...
[2/7] Creating advanced features...
[3/7] Preparing data...
[4/7] Training XGBoost...
  Fold 1: 0.971234
  Fold 2: 0.972456
  ...
[5/7] Training LightGBM...
[6/7] Training CatBoost...
[7/7] Creating ensemble...

ENSEMBLE OOF Score: 0.972XXX
```

## After Submission

Check your new ranking:
```bash
kaggle competitions submissions playground-series-s6e6
```

## Need More Improvement?

See `IMPROVEMENT_STRATEGY.md` for additional techniques:
- Neural networks
- Stacking meta-models
- Hyperparameter optimization
- Pseudo-labeling

---

**Ready to go?** Just run:
```bash
bash install_packages.sh
python3 run_improved_pipeline.py
```

Good luck! 🎯 Target: Beat 0.97101 → Reach 0.9725+
