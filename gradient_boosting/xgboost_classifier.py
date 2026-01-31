"""
Gradient Boosting Classifier for 16 Personality Types (MBTI)
============================================================

This script trains an XGBoost classifier on the 16P personality dataset
with a 70/15/15 train/validation/test split and provides detailed evaluation.

All operations use fixed random seeds for reproducibility.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    top_k_accuracy_score
)
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION - All random seeds for reproducibility
# =============================================================================
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Data splits
TEST_SIZE = 0.15      # 15% for test
VAL_SIZE = 0.176      # 15% of remaining 85% ≈ 15% of total

# XGBoost hyperparameters
XGBOOST_PARAMS = {
    'n_estimators': 500,
    'learning_rate': 0.1,
    'max_depth': 6,
    'min_child_weight': 1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': 'multi:softprob',
    'num_class': 16,
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    'verbosity': 0
}

EARLY_STOPPING_ROUNDS = 15

# =============================================================================
# STEP 1: LOAD DATA
# =============================================================================
print("=" * 80)
print("GRADIENT BOOSTING CLASSIFIER FOR 16 PERSONALITY TYPES")
print("=" * 80)

print("\n[STEP 1] Loading data...")
df = pd.read_csv('16P_eda_cleaned.csv')
print(f"  -> Loaded {len(df):,} rows and {len(df.columns)} columns")

# Separate features and target
X = df.drop(columns=['Personality'])
y = df['Personality']

feature_columns = X.columns.tolist()
print(f"  -> Features: {len(feature_columns)} survey questions")
print(f"  -> Target: 16 MBTI personality types")

# =============================================================================
# STEP 2: ENCODE TARGET LABELS
# =============================================================================
print("\n[STEP 2] Encoding target labels...")
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
class_names = label_encoder.classes_
print(f"  -> Classes: {', '.join(class_names)}")

# =============================================================================
# STEP 3: SPLIT DATA (70% Train, 15% Validation, 15% Test)
# =============================================================================
print("\n[STEP 3] Splitting data (70/15/15)...")

# First split: 85% train+val, 15% test
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y_encoded, 
    test_size=TEST_SIZE, 
    random_state=RANDOM_STATE, 
    stratify=y_encoded
)

# Second split: 70% train, 15% val (of total)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, 
    test_size=VAL_SIZE, 
    random_state=RANDOM_STATE, 
    stratify=y_temp
)

print(f"  -> Training set:   {len(X_train):,} samples ({len(X_train)/len(df)*100:.1f}%)")
print(f"  -> Validation set: {len(X_val):,} samples ({len(X_val)/len(df)*100:.1f}%)")
print(f"  -> Test set:       {len(X_test):,} samples ({len(X_test)/len(df)*100:.1f}%)")

# Verify stratification
print("\n  Stratification check (samples per class in each set):")
for i, name in enumerate(class_names[:4]):  # Show first 4 as sample
    train_count = (y_train == i).sum()
    val_count = (y_val == i).sum()
    test_count = (y_test == i).sum()
    print(f"    {name}: Train={train_count}, Val={val_count}, Test={test_count}")
print("    ...")

# =============================================================================
# STEP 4: TRAIN XGBOOST MODEL
# =============================================================================
print("\n[STEP 4] Training XGBoost model...")
print(f"  -> n_estimators: {XGBOOST_PARAMS['n_estimators']}")
print(f"  -> learning_rate: {XGBOOST_PARAMS['learning_rate']}")
print(f"  -> max_depth: {XGBOOST_PARAMS['max_depth']}")
print(f"  -> early_stopping_rounds: {EARLY_STOPPING_ROUNDS}")

model = XGBClassifier(**XGBOOST_PARAMS, early_stopping_rounds=EARLY_STOPPING_ROUNDS)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

best_iteration = model.best_iteration
print(f"\n  -> Training complete!")
print(f"  -> Best iteration: {best_iteration} (stopped early from {XGBOOST_PARAMS['n_estimators']})")

# =============================================================================
# STEP 5: PREDICTIONS
# =============================================================================
print("\n[STEP 5] Making predictions...")

y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)
y_test_pred = model.predict(X_test)

# Probability predictions for top-k accuracy
y_test_proba = model.predict_proba(X_test)

# =============================================================================
# STEP 6: DETAILED EVALUATION
# =============================================================================
print("\n" + "=" * 80)
print("DETAILED EVALUATION RESULTS")
print("=" * 80)

# 6.1 Overall Accuracy
print("\n[6.1] ACCURACY SCORES")
print("-" * 40)
train_acc = accuracy_score(y_train, y_train_pred)
val_acc = accuracy_score(y_val, y_val_pred)
test_acc = accuracy_score(y_test, y_test_pred)

print(f"  Training Accuracy:   {train_acc:.4f} ({train_acc*100:.2f}%)")
print(f"  Validation Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
print(f"  Test Accuracy:       {test_acc:.4f} ({test_acc*100:.2f}%)")

# Check for overfitting
overfit_gap = train_acc - test_acc
if overfit_gap > 0.05:
    print(f"\n  ⚠️  Potential overfitting detected (train-test gap: {overfit_gap:.2%})")
else:
    print(f"\n  ✓ Model generalizes well (train-test gap: {overfit_gap:.2%})")

# 6.2 Top-K Accuracy
print("\n[6.2] TOP-K ACCURACY (Test Set)")
print("-" * 40)
for k in [1, 2, 3, 5]:
    if k <= len(class_names):
        top_k_acc = top_k_accuracy_score(y_test, y_test_proba, k=k)
        print(f"  Top-{k} Accuracy: {top_k_acc:.4f} ({top_k_acc*100:.2f}%)")

# 6.3 Macro & Weighted Metrics
print("\n[6.3] AGGREGATE METRICS (Test Set)")
print("-" * 40)
print(f"  Macro Precision:    {precision_score(y_test, y_test_pred, average='macro'):.4f}")
print(f"  Macro Recall:       {recall_score(y_test, y_test_pred, average='macro'):.4f}")
print(f"  Macro F1-Score:     {f1_score(y_test, y_test_pred, average='macro'):.4f}")
print(f"  Weighted F1-Score:  {f1_score(y_test, y_test_pred, average='weighted'):.4f}")

# 6.4 Per-Class Classification Report
print("\n[6.4] PER-CLASS CLASSIFICATION REPORT (Test Set)")
print("-" * 80)
print(classification_report(y_test, y_test_pred, target_names=class_names, digits=4))

# 6.5 Confusion Matrix
print("\n[6.5] CONFUSION MATRIX (Test Set)")
print("-" * 40)
cm = confusion_matrix(y_test, y_test_pred)
print("  (Saved as 'xgb_confusion_matrix.png')")

# 6.6 Feature Importance
print("\n[6.6] TOP 15 MOST IMPORTANT FEATURES")
print("-" * 60)
feature_importance = model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': feature_importance
}).sort_values('Importance', ascending=False)

for i, row in importance_df.head(15).iterrows():
    feature_short = row['Feature'][:55] + "..." if len(row['Feature']) > 55 else row['Feature']
    print(f"  {importance_df.head(15).index.get_loc(i)+1:2}. [{row['Importance']:.4f}] {feature_short}")

# =============================================================================
# STEP 7: SAVE VISUALIZATIONS
# =============================================================================
print("\n[STEP 7] Saving visualizations...")

# Create output directory for figures
os.makedirs('figures', exist_ok=True)

# 7.1 Confusion Matrix Heatmap
plt.figure(figsize=(14, 12))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix - Gradient Boosting (16 Personality Types)', fontsize=14)
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('Actual', fontsize=12)
plt.tight_layout()
plt.savefig('figures/xgb_confusion_matrix.png', dpi=150)
plt.close()
print("  -> Saved: figures/xgb_confusion_matrix.png")

# 7.2 Feature Importance Plot
plt.figure(figsize=(12, 10))
top_20 = importance_df.head(20)
short_names = [f[:40] + "..." if len(f) > 40 else f for f in top_20['Feature']]
plt.barh(range(len(top_20)), top_20['Importance'].values, color='steelblue')
plt.yticks(range(len(top_20)), short_names)
plt.xlabel('Importance Score', fontsize=12)
plt.title('Top 20 Most Important Features - Gradient Boosting', fontsize=14)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('figures/xgb_feature_importance.png', dpi=150)
plt.close()
print("  -> Saved: figures/xgb_feature_importance.png")

# 7.3 Per-Class Accuracy Bar Chart
per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
plt.figure(figsize=(12, 6))
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(class_names)))
bars = plt.bar(class_names, per_class_accuracy, color=colors)
plt.axhline(y=test_acc, color='red', linestyle='--', label=f'Overall Accuracy: {test_acc:.2%}')
plt.xlabel('Personality Type', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Per-Class Accuracy - Gradient Boosting', fontsize=14)
plt.xticks(rotation=45)
plt.legend()
plt.ylim(0, 1)
for bar, acc in zip(bars, per_class_accuracy):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{acc:.1%}', ha='center', va='bottom', fontsize=8)
plt.tight_layout()
plt.savefig('figures/xgb_per_class_accuracy.png', dpi=150)
plt.close()
print("  -> Saved: figures/xgb_per_class_accuracy.png")

# =============================================================================
# STEP 8: SAVE DETAILED RESULTS TO FILE
# =============================================================================
print("\n[STEP 8] Saving detailed results to file...")

with open('xgb_evaluation_report.txt', 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("GRADIENT BOOSTING EVALUATION REPORT\n")
    f.write("16 Personality Types (MBTI) Classification\n")
    f.write("=" * 80 + "\n\n")
    
    f.write("CONFIGURATION\n")
    f.write("-" * 40 + "\n")
    f.write(f"Random State: {RANDOM_STATE}\n")
    f.write(f"Data Split: 70% Train / 15% Validation / 15% Test\n")
    f.write(f"Total Samples: {len(df):,}\n")
    f.write(f"Training Samples: {len(X_train):,}\n")
    f.write(f"Validation Samples: {len(X_val):,}\n")
    f.write(f"Test Samples: {len(X_test):,}\n\n")
    
    f.write("XGBOOST HYPERPARAMETERS\n")
    f.write("-" * 40 + "\n")
    for key, value in XGBOOST_PARAMS.items():
        f.write(f"{key}: {value}\n")
    f.write(f"early_stopping_rounds: {EARLY_STOPPING_ROUNDS}\n")
    f.write(f"best_iteration: {best_iteration}\n\n")
    
    f.write("ACCURACY SCORES\n")
    f.write("-" * 40 + "\n")
    f.write(f"Training Accuracy:   {train_acc:.4f} ({train_acc*100:.2f}%)\n")
    f.write(f"Validation Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)\n")
    f.write(f"Test Accuracy:       {test_acc:.4f} ({test_acc*100:.2f}%)\n\n")
    
    f.write("TOP-K ACCURACY (Test Set)\n")
    f.write("-" * 40 + "\n")
    for k in [1, 2, 3, 5]:
        top_k_acc = top_k_accuracy_score(y_test, y_test_proba, k=k)
        f.write(f"Top-{k} Accuracy: {top_k_acc:.4f} ({top_k_acc*100:.2f}%)\n")
    f.write("\n")
    
    f.write("CLASSIFICATION REPORT (Test Set)\n")
    f.write("-" * 80 + "\n")
    f.write(classification_report(y_test, y_test_pred, target_names=class_names, digits=4))
    f.write("\n")
    
    f.write("TOP 20 IMPORTANT FEATURES\n")
    f.write("-" * 80 + "\n")
    for i, (_, row) in enumerate(importance_df.head(20).iterrows()):
        f.write(f"{i+1:2}. [{row['Importance']:.4f}] {row['Feature']}\n")

print("  -> Saved: xgb_evaluation_report.txt")

# Save feature importance to CSV
importance_df.to_csv('xgb_feature_importance.csv', index=False)
print("  -> Saved: xgb_feature_importance.csv")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"""
Model: XGBoost Gradient Boosting Classifier
Dataset: 16P_eda_cleaned.csv ({len(df):,} samples)
Split: 70% Train / 15% Validation / 15% Test

RESULTS:
  • Test Accuracy: {test_acc:.2%}
  • Top-3 Accuracy: {top_k_accuracy_score(y_test, y_test_proba, k=3):.2%}
  • Macro F1-Score: {f1_score(y_test, y_test_pred, average='macro'):.4f}
  • Best Iteration: {best_iteration} trees

OUTPUT FILES:
  • figures/xgb_confusion_matrix.png
  • figures/xgb_feature_importance.png
  • figures/xgb_per_class_accuracy.png
  • xgb_evaluation_report.txt
  • xgb_feature_importance.csv

Reproducibility: All operations used random_state={RANDOM_STATE}
""")
print("=" * 80)
