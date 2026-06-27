# Predicting Stellar Class - Kaggle Playground Series S6E6

## Competition Overview
- **Competition**: Playground Series S6E6 - Predicting Stellar Class
- **URL**: https://www.kaggle.com/competitions/playground-series-s6e6
- **Metric**: Balanced Accuracy
- **Task**: Multi-class classification (GALAXY, QSO, STAR)

## Dataset
- **Train**: 577,347 rows
- **Test**: 577,351 rows
- **Features**: 
  - `alpha`: Right ascension angle
  - `delta`: Declination angle
  - `u, g, r, i, z`: Five photometric filters (magnitudes)
  - `redshift`: Redshift value
  - `spectral_type`: Spectral type (O/B, A/F, G/K, M)
  - `galaxy_population`: Galaxy population (Red_Sequence, Blue_Cloud, Green_Valley)
- **Target**: `class` (GALAXY, QSO, STAR)

## Files in This Directory
1. **stellar_class_submission.ipynb** - Main submission notebook with XGBoost baseline
2. **gpu-logistic-regression-stacker.ipynb** - Advanced ensemble stacker example
3. **train.csv** - Training data
4. **test.csv** - Test data
5. **sample_submission.csv** - Submission format example

## Quick Start

### 1. Run the Baseline Notebook
Open `stellar_class_submission.ipynb` in Jupyter and run all cells:

```bash
jupyter notebook stellar_class_submission.ipynb
```

### 2. Submit to Kaggle
After running the notebook, submit your results:

```bash
kaggle competitions submit -c playground-series-s6e6 -f submission.csv -m "XGBoost baseline submission"
```

### 3. Check Your Submission
```bash
kaggle competitions submissions playground-series-s6e6
```

## Baseline Model
The baseline notebook uses:
- **Algorithm**: XGBoost
- **Cross-validation**: 5-fold Stratified K-Fold
- **Features**: All available features (alpha, delta, u, g, r, i, z, redshift, spectral_type, galaxy_population)
- **Expected CV Score**: ~0.96-0.97 balanced accuracy

## Advanced Approach (Optional)
The `gpu-logistic-regression-stacker.ipynb` shows an advanced ensemble approach:
- Combines predictions from multiple models (XGBoost, CatBoost, Neural Networks, etc.)
- Uses GPU-accelerated logistic regression for stacking
- Achieves higher scores through ensemble diversity

## Next Steps for Improvement
1. **Feature Engineering**:
   - Color indices (e.g., u-g, g-r, r-i)
   - Polynomial features
   - Interaction terms between spectral_type and photometric bands

2. **Model Tuning**:
   - Hyperparameter optimization (Optuna, GridSearchCV)
   - Try different algorithms (LightGBM, CatBoost, Neural Networks)

3. **Ensemble Methods**:
   - Blend multiple models
   - Use stacking like the advanced notebook

4. **Data Analysis**:
   - Check for outliers
   - Analyze class separability in feature space
   - Visualize decision boundaries

## Competition Tips
- The metric is **balanced accuracy**, so focus on per-class performance
- The dataset is balanced, so standard accuracy and balanced accuracy should be similar
- Cross-validation scores correlate well with leaderboard scores
- Try to understand the physical meaning of features (astronomy domain knowledge helps!)

## Resources
- Competition Discussion: https://www.kaggle.com/competitions/playground-series-s6e6/discussion
- Example Notebooks: https://www.kaggle.com/competitions/playground-series-s6e6/code
- Your Kaggle Profile: https://www.kaggle.com/dgi1995

Good luck! 🚀
