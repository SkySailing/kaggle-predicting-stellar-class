#!/usr/bin/env python3
"""
Improved Stellar Class Prediction Pipeline
Expected improvement: +0.002-0.003 over baseline (0.97101 -> 0.9725+)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)

print("="*70)
print("STELLAR CLASS PREDICTION - IMPROVED PIPELINE")
print("="*70)

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n[1/7] Loading data...")

# Handle different data paths (local vs Kaggle notebook)
import os
try:
    from config import LOCAL_DATA_PATH, KAGGLE_DATA_PATH, CURRENT_DIR_PATH
except ImportError:
    LOCAL_DATA_PATH = "Data/playground-series-s6e6/"
    KAGGLE_DATA_PATH = "/kaggle/input/playground-series-s6e6/"
    CURRENT_DIR_PATH = ""

# Try different paths
if os.path.exists(LOCAL_DATA_PATH + 'train.csv'):
    data_path = LOCAL_DATA_PATH
    print(f"Using local data path: {data_path}")
elif os.path.exists(KAGGLE_DATA_PATH + 'train.csv'):
    data_path = KAGGLE_DATA_PATH
    print(f"Using Kaggle notebook path: {data_path}")
else:
    data_path = CURRENT_DIR_PATH
    print("Using current directory for data")

train = pd.read_csv(f'{data_path}train.csv')
test = pd.read_csv(f'{data_path}test.csv')
sample_submission = pd.read_csv(f'{data_path}sample_submission.csv')

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"\nClass distribution:")
print(train['class'].value_counts(normalize=True))

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================
print("\n[2/7] Creating advanced features...")

def create_features(df):
    """Create advanced astronomical features"""
    df = df.copy()

    # 1. Color indices (very important in astronomy!)
    df['u-g'] = df['u'] - df['g']
    df['g-r'] = df['g'] - df['r']
    df['r-i'] = df['r'] - df['i']
    df['i-z'] = df['i'] - df['z']
    df['u-r'] = df['u'] - df['r']
    df['g-i'] = df['g'] - df['i']
    df['r-z'] = df['r'] - df['z']
    df['u-z'] = df['u'] - df['z']

    # 2. Color ratios
    df['color_ratio_1'] = (df['u'] - df['g']) / (df['g'] - df['r'] + 1e-5)
    df['color_ratio_2'] = (df['g'] - df['r']) / (df['r'] - df['i'] + 1e-5)
    df['color_ratio_3'] = (df['r'] - df['i']) / (df['i'] - df['z'] + 1e-5)

    # 3. Magnitude statistics
    df['mag_mean'] = df[['u', 'g', 'r', 'i', 'z']].mean(axis=1)
    df['mag_std'] = df[['u', 'g', 'r', 'i', 'z']].std(axis=1)
    df['mag_min'] = df[['u', 'g', 'r', 'i', 'z']].min(axis=1)
    df['mag_max'] = df[['u', 'g', 'r', 'i', 'z']].max(axis=1)
    df['mag_range'] = df['mag_max'] - df['mag_min']

    # 4. Redshift interactions
    df['redshift_squared'] = df['redshift'] ** 2
    df['redshift_log'] = np.log1p(df['redshift'])
    df['redshift_x_mag'] = df['redshift'] * df['mag_mean']

    # 5. Position features
    df['alpha_sin'] = np.sin(df['alpha'] * np.pi / 180)
    df['alpha_cos'] = np.cos(df['alpha'] * np.pi / 180)
    df['delta_sin'] = np.sin(df['delta'] * np.pi / 180)
    df['delta_cos'] = np.cos(df['delta'] * np.pi / 180)

    # 6. Photometric bands ratios
    for band in ['u', 'g', 'r', 'i', 'z']:
        df[f'{band}_ratio_mean'] = df[band] / (df['mag_mean'] + 1e-5)

    return df

train_fe = create_features(train)
test_fe = create_features(test)

print(f"Original features: {train.shape[1]}")
print(f"After feature engineering: {train_fe.shape[1]}")

# ============================================================================
# PREPARE DATA
# ============================================================================
print("\n[3/7] Preparing data...")

# Encode categorical features
le_spectral = LabelEncoder()
le_galaxy = LabelEncoder()

train_fe['spectral_type_encoded'] = le_spectral.fit_transform(train_fe['spectral_type'])
test_fe['spectral_type_encoded'] = le_spectral.transform(test_fe['spectral_type'])

train_fe['galaxy_population_encoded'] = le_galaxy.fit_transform(train_fe['galaxy_population'])
test_fe['galaxy_population_encoded'] = le_galaxy.transform(test_fe['galaxy_population'])

# Target encoding
target_map = {'GALAXY': 0, 'QSO': 1, 'STAR': 2}
inv_target_map = {v: k for k, v in target_map.items()}

# Define features
exclude_cols = ['id', 'class', 'spectral_type', 'galaxy_population']
feature_cols = [col for col in train_fe.columns if col not in exclude_cols]

X_train = train_fe[feature_cols]
y_train = train_fe['class'].map(target_map)
X_test = test_fe[feature_cols]

print(f"Features used: {len(feature_cols)}")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")

# ============================================================================
# MODEL 1: XGBoost
# ============================================================================
print("\n[4/7] Training XGBoost...")

xgb_params = {
    'objective': 'multi:softprob',
    'num_class': 3,
    'eval_metric': 'mlogloss',
    'max_depth': 10,
    'learning_rate': 0.02,
    'subsample': 0.85,
    'colsample_bytree': 0.85,
    'min_child_weight': 1,
    'gamma': 0.1,
    'reg_alpha': 0.5,
    'reg_lambda': 2.0,
    'seed': SEED,
    'n_jobs': -1,
    'tree_method': 'hist'
}

skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=SEED)
xgb_oof = np.zeros((len(X_train), 3))
xgb_test = np.zeros((len(X_test), 3))
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    model = xgb.XGBClassifier(**xgb_params, n_estimators=1000, early_stopping_rounds=50)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    val_preds = model.predict_proba(X_val)
    xgb_oof[val_idx] = val_preds
    xgb_test += model.predict_proba(X_test) / 5

    fold_score = balanced_accuracy_score(y_val, np.argmax(val_preds, axis=1))
    cv_scores.append(fold_score)
    print(f"  Fold {fold}: {fold_score:.6f}")

xgb_score = balanced_accuracy_score(y_train, np.argmax(xgb_oof, axis=1))
print(f"\nXGBoost OOF Score: {xgb_score:.6f}")

# ============================================================================
# MODEL 2: LightGBM
# ============================================================================
print("\n[5/7] Training LightGBM...")

lgb_params = {
    'objective': 'multiclass',
    'num_class': 3,
    'metric': 'multi_logloss',
    'boosting_type': 'gbdt',
    'learning_rate': 0.02,
    'num_leaves': 127,
    'max_depth': 10,
    'min_child_samples': 20,
    'subsample': 0.85,
    'colsample_bytree': 0.85,
    'reg_alpha': 0.5,
    'reg_lambda': 2.0,
    'seed': SEED,
    'n_jobs': -1,
    'verbose': -1
}

skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=SEED+1)
lgb_oof = np.zeros((len(X_train), 3))
lgb_test = np.zeros((len(X_test), 3))
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    train_data = lgb.Dataset(X_tr, label=y_tr)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    model = lgb.train(
        lgb_params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )

    val_preds = model.predict(X_val, num_iteration=model.best_iteration)
    lgb_oof[val_idx] = val_preds
    lgb_test += model.predict(X_test, num_iteration=model.best_iteration) / 5

    fold_score = balanced_accuracy_score(y_val, np.argmax(val_preds, axis=1))
    cv_scores.append(fold_score)
    print(f"  Fold {fold}: {fold_score:.6f}")

lgb_score = balanced_accuracy_score(y_train, np.argmax(lgb_oof, axis=1))
print(f"\nLightGBM OOF Score: {lgb_score:.6f}")

# ============================================================================
# MODEL 3: CatBoost
# ============================================================================
print("\n[6/7] Training CatBoost...")

cat_params = {
    'iterations': 1000,
    'learning_rate': 0.02,
    'depth': 10,
    'l2_leaf_reg': 3.0,
    'random_strength': 1.0,
    'bagging_temperature': 0.5,
    'od_type': 'Iter',
    'od_wait': 50,
    'random_seed': SEED,
    'verbose': False,
    'loss_function': 'MultiClass',
    'eval_metric': 'MultiClass'
}

skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=SEED+2)
cat_oof = np.zeros((len(X_train), 3))
cat_test = np.zeros((len(X_test), 3))
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    model = CatBoostClassifier(**cat_params)
    model.fit(
        X_tr, y_tr,
        eval_set=(X_val, y_val),
        verbose=False
    )

    val_preds = model.predict_proba(X_val)
    cat_oof[val_idx] = val_preds
    cat_test += model.predict_proba(X_test) / 5

    fold_score = balanced_accuracy_score(y_val, np.argmax(val_preds, axis=1))
    cv_scores.append(fold_score)
    print(f"  Fold {fold}: {fold_score:.6f}")

cat_score = balanced_accuracy_score(y_train, np.argmax(cat_oof, axis=1))
print(f"\nCatBoost OOF Score: {cat_score:.6f}")

# ============================================================================
# ENSEMBLE
# ============================================================================
print("\n[7/7] Creating ensemble...")

# Calculate optimal weights based on OOF scores
scores = np.array([xgb_score, lgb_score, cat_score])
weights = scores / scores.sum()

print("\nEnsemble Weights:")
print(f"  XGBoost:  {weights[0]:.4f} (score: {xgb_score:.6f})")
print(f"  LightGBM: {weights[1]:.4f} (score: {lgb_score:.6f})")
print(f"  CatBoost: {weights[2]:.4f} (score: {cat_score:.6f})")

# Weighted ensemble
ensemble_oof = (xgb_oof * weights[0] +
                lgb_oof * weights[1] +
                cat_oof * weights[2])

ensemble_test = (xgb_test * weights[0] +
                 lgb_test * weights[1] +
                 cat_test * weights[2])

ensemble_score = balanced_accuracy_score(y_train, np.argmax(ensemble_oof, axis=1))

print(f"\n{'='*70}")
print(f"ENSEMBLE OOF Score: {ensemble_score:.6f}")
print(f"{'='*70}")
print(f"Improvement over best single model: +{ensemble_score - max(xgb_score, lgb_score, cat_score):.6f}")

# Per-class accuracy
ensemble_pred_labels = np.argmax(ensemble_oof, axis=1)
print("\nPer-class Accuracy:")
for i, class_name in enumerate(['GALAXY', 'QSO', 'STAR']):
    mask = (y_train == i)
    class_acc = (ensemble_pred_labels[mask] == i).mean()
    print(f"  {class_name:>6s}: {class_acc:.6f}")

# ============================================================================
# CREATE SUBMISSION
# ============================================================================
print("\nCreating submission...")

test_pred_labels = np.argmax(ensemble_test, axis=1)
submission = sample_submission.copy()
submission['class'] = test_pred_labels
submission['class'] = submission['class'].map(inv_target_map)

submission.to_csv('submission_improved.csv', index=False)

print("\n" + "="*70)
print("SUBMISSION CREATED: submission_improved.csv")
print("="*70)
print(f"\nExpected LB score: ~{ensemble_score:.5f}")
print(f"Your current score: 0.97101")
print(f"Expected improvement: +{ensemble_score - 0.97101:.5f}")
print(f"\nTo submit:")
print("  kaggle competitions submit -c playground-series-s6e6 \\")
print("    -f submission_improved.csv \\")
print("    -m 'Ensemble: XGB+LGB+CAT with advanced features'")
print("\nSubmission class distribution:")
print(submission['class'].value_counts(normalize=True))
