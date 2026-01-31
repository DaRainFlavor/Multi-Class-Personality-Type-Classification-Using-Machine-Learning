"""
Random Forest Classifier for 16 Personality Types (MBTI)
=========================================================

This script trains a Random Forest classifier on the 16P personality dataset
with a 70/15/15 train/validation/test split and provides detailed evaluation.

All operations use fixed random seeds for reproducibility.
Uses the SAME data splits as other models (Gradient Boosting, Logistic Regression, LDA) for fair comparison.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    top_k_accuracy_score
)
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

# Data splits (SAME as other models for fair comparison)
TEST_SIZE = 0.15      # 15% for test
VAL_SIZE = 0.176      # 15% of remaining 85% ≈ 15% of total

# Random Forest hyperparameters
RF_PARAMS = {
    'n_estimators': 100,          # Number of trees in the forest
    'max_depth': 20,              # Maximum depth of trees
    'min_samples_split': 5,       # Minimum samples required to split
    'min_samples_leaf': 2,        # Minimum samples required at leaf node
    'max_features': 'sqrt',       # Number of features to consider at each split
    'random_state': RANDOM_STATE,
    'n_jobs': -1,                 # Use all available cores
    'verbose': 1                  # Show progress
}

# =============================================================================
# STEP 1: LOAD DATA
# =============================================================================
print("=" * 80)
print("RANDOM FOREST CLASSIFIER FOR 16 PERSONALITY TYPES")
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
# STEP 4: TRAIN RANDOM FOREST MODEL
# =============================================================================
print("\n[STEP 4] Training Random Forest model...")
print(f"  -> n_estimators: {RF_PARAMS['n_estimators']}")
print(f"  -> max_depth: {RF_PARAMS['max_depth']}")
print(f"  -> max_features: {RF_PARAMS['max_features']}")
print(f"  -> min_samples_split: {RF_PARAMS['min_samples_split']}")

model = RandomForestClassifier(**RF_PARAMS)

model.fit(X_train, y_train)

print(f"\n  -> Training complete!")
print(f"  -> Number of trees: {len(model.estimators_)}")

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
    print(f"\n  [!] Potential overfitting detected (train-test gap: {overfit_gap:.2%})")
else:
    print(f"\n  [OK] Model generalizes well (train-test gap: {overfit_gap:.2%})")

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
print("  (Saved as 'rf_confusion_matrix.png')")

# 6.6 Feature Importance
print("\n[6.6] TOP 15 MOST IMPORTANT FEATURES (by Random Forest importance)")
print("-" * 60)

importance_df = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': model.feature_importances_
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
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix - Random Forest (16 Personality Types)', fontsize=14)
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('Actual', fontsize=12)
plt.tight_layout()
plt.savefig('figures/rf_confusion_matrix.png', dpi=150)
plt.close()
print("  -> Saved: figures/rf_confusion_matrix.png")

# 7.2 Feature Importance Plot
plt.figure(figsize=(12, 10))
top_20 = importance_df.head(20)
short_names = [f[:40] + "..." if len(f) > 40 else f for f in top_20['Feature']]
plt.barh(range(len(top_20)), top_20['Importance'].values, color='forestgreen')
plt.yticks(range(len(top_20)), short_names)
plt.xlabel('Feature Importance', fontsize=12)
plt.title('Top 20 Most Important Features - Random Forest', fontsize=14)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('figures/rf_feature_importance.png', dpi=150)
plt.close()
print("  -> Saved: figures/rf_feature_importance.png")

# 7.3 Per-Class Accuracy Bar Chart
per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
plt.figure(figsize=(12, 6))
colors = plt.cm.Greens(np.linspace(0.4, 0.9, len(class_names)))
bars = plt.bar(class_names, per_class_accuracy, color=colors)
plt.axhline(y=test_acc, color='red', linestyle='--', label=f'Overall Accuracy: {test_acc:.2%}')
plt.xlabel('Personality Type', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Per-Class Accuracy - Random Forest', fontsize=14)
plt.xticks(rotation=45)
plt.legend()
plt.ylim(0, 1)
for bar, acc in zip(bars, per_class_accuracy):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{acc:.1%}', ha='center', va='bottom', fontsize=8)
plt.tight_layout()
plt.savefig('figures/rf_per_class_accuracy.png', dpi=150)
plt.close()
print("  -> Saved: figures/rf_per_class_accuracy.png")

# =============================================================================
# STEP 8: SAVE DETAILED RESULTS TO FILE
# =============================================================================
print("\n[STEP 8] Saving detailed results to file...")

with open('rf_evaluation_report.txt', 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("RANDOM FOREST EVALUATION REPORT\n")
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
    
    f.write("RANDOM FOREST HYPERPARAMETERS\n")
    f.write("-" * 40 + "\n")
    for key, value in RF_PARAMS.items():
        if key != 'verbose':  # Skip verbose in report
            f.write(f"{key}: {value}\n")
    f.write(f"n_trees (actual): {len(model.estimators_)}\n\n")
    
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
    
    f.write("TOP 20 IMPORTANT FEATURES (by Random Forest importance)\n")
    f.write("-" * 80 + "\n")
    for i, (_, row) in enumerate(importance_df.head(20).iterrows()):
        f.write(f"{i+1:2}. [{row['Importance']:.4f}] {row['Feature']}\n")

print("  -> Saved: rf_evaluation_report.txt")

# Save feature importance to CSV
importance_df.to_csv('rf_feature_importance.csv', index=False)
print("  -> Saved: rf_feature_importance.csv")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"""
Model: Random Forest Classifier
Dataset: 16P_eda_cleaned.csv ({len(df):,} samples)
Split: 70% Train / 15% Validation / 15% Test

RESULTS:
  • Test Accuracy: {test_acc:.2%}
  • Top-3 Accuracy: {top_k_accuracy_score(y_test, y_test_proba, k=3):.2%}
  • Macro F1-Score: {f1_score(y_test, y_test_pred, average='macro'):.4f}
  • Number of Trees: {len(model.estimators_)}

OUTPUT FILES:
  • figures/rf_confusion_matrix.png
  • figures/rf_feature_importance.png
  • figures/rf_per_class_accuracy.png
  • rf_evaluation_report.txt
  • rf_feature_importance.csv

Reproducibility: All operations used random_state={RANDOM_STATE}
NOTE: Uses SAME data splits as Gradient Boosting, Logistic Regression, and LDA for fair comparison
""")
print("=" * 80)
