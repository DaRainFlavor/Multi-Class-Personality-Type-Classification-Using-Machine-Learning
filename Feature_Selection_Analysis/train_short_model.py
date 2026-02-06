"""
XGBoost Feature Selection and Short Model Training Script

This script:
1. Uses the EXACT same data split as Feature_Ranking_Analysis.ipynb (70/15/15)
2. Trains XGBoost to get feature importance
3. Extracts top 35 features
4. Trains a new XGBoost model with only top 35 features
5. Reports accuracy comparison
6. Exports ONNX model and question list for website deployment
"""

import pandas as pd
import numpy as np
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import joblib

# For ONNX conversion
try:
    from skl2onnx import convert_sklearn, update_registered_converter
    from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes
    from onnxmltools.convert.xgboost.operator_converters.XGBClassifier import convert_xgboost
    from skl2onnx.common.data_types import FloatTensorType
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("Warning: ONNX conversion libraries not available. Will save joblib model only.")

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_PATH = os.path.join(PARENT_DIR, '16P_eda_cleaned.csv')
API_DIR = os.path.join(PARENT_DIR, 'mbti-quiz', 'api')

# XGBoost parameters (same as ML_Comparison_Analysis.ipynb and Feature_Ranking_Analysis.ipynb)
XGBOOST_PARAMS = {
    'n_estimators': 500,
    'learning_rate': 0.1,
    'max_depth': 6,
    'min_child_weight': 1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': 'multi:softprob',
    'random_state': 42,
    'n_jobs': -1,
    'verbosity': 0
}

# Number of top features to select
TOP_N_FEATURES = 35


def load_and_preprocess_data():
    """Load and preprocess the dataset exactly as in Feature_Ranking_Analysis.ipynb"""
    print(f"Loading data from: {DATA_PATH}")
    
    # Try different encodings
    try:
        df = pd.read_csv(DATA_PATH)
    except UnicodeDecodeError:
        df = pd.read_csv(DATA_PATH, encoding='cp1252')
    
    print(f"Dataset shape: {df.shape}")
    
    # Drop identifier if exists
    if 'Response Id' in df.columns:
        df = df.drop(columns=['Response Id'])
    
    # Identify columns to drop (derived scores, etc.)
    cols_to_drop = ['Personality', 'E_I_score', 'S_N_score', 'T_F_score', 'J_P_score', 
                    'E_I_strength', 'S_N_strength', 'T_F_strength', 'J_P_strength',
                    'is_Extraverted', 'is_Intuitive', 'is_Feeling', 'is_Judging', 
                    'Consistency', 'Original_Personality']
    
    # Filter columns that actually exist in the dataframe
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    
    X = df.drop(columns=cols_to_drop)
    y = df['Personality']
    
    feature_names = X.columns.tolist()
    print(f"Features (Questions): {len(feature_names)}")
    print(f"Target Classes: {y.nunique()}")
    
    return X, y, feature_names


def split_data(X, y):
    """
    Split data EXACTLY as in Feature_Ranking_Analysis.ipynb:
    70% Train, 15% Validation, 15% Test
    """
    # Encode Target
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Split parameters from Feature_Ranking_Analysis.ipynb
    TEST_SIZE = 0.15      # 15% for test
    VAL_SIZE = 0.176      # 15% of remaining 85% ≈ 15% of total (0.15/0.85 ≈ 0.176)
    
    # First split: 85% train+val, 15% test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y_encoded, 
        test_size=TEST_SIZE, 
        random_state=42, 
        stratify=y_encoded
    )
    
    # Second split: 70% train, 15% val (of total)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, 
        test_size=VAL_SIZE, 
        random_state=42, 
        stratify=y_temp
    )
    
    print(f"Training set:   {len(X_train):,} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"Validation set: {len(X_val):,} samples ({len(X_val)/len(X)*100:.1f}%)")
    print(f"Test set:       {len(X_test):,} samples ({len(X_test)/len(X)*100:.1f}%)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, le


def train_full_model_and_get_importance(X_train, X_val, y_train, y_val, feature_names, le):
    """Train XGBoost on all features and extract feature importance"""
    print("\n" + "="*60)
    print("STEP 1: Training XGBoost on ALL 60 features")
    print("="*60)
    
    params = XGBOOST_PARAMS.copy()
    params['num_class'] = len(le.classes_)
    
    model = XGBClassifier(**params, early_stopping_rounds=15)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    # Extract feature importance
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False).reset_index(drop=True)
    
    print("\nTop 10 Most Important Features:")
    print(feature_importance_df.head(10).to_string())
    
    return model, feature_importance_df


def train_short_model(X_train, X_val, X_test, y_train, y_val, y_test, 
                      top_features, le, feature_names):
    """Train XGBoost on only the top N features"""
    print(f"\n" + "="*60)
    print(f"STEP 2: Training XGBoost on TOP {TOP_N_FEATURES} features")
    print("="*60)
    
    # Get indices of top features in original order
    top_feature_indices = []
    for feat in top_features:
        idx = feature_names.index(feat)
        top_feature_indices.append(idx)
    
    # Subset data to only top features
    X_train_short = X_train[top_features]
    X_val_short = X_val[top_features]
    X_test_short = X_test[top_features]
    
    params = XGBOOST_PARAMS.copy()
    params['num_class'] = len(le.classes_)
    
    model_short = XGBClassifier(**params, early_stopping_rounds=15)
    model_short.fit(X_train_short, y_train, eval_set=[(X_val_short, y_val)], verbose=False)
    
    return model_short, top_feature_indices


def evaluate_models(full_model, short_model, X_test, y_test, top_features, le):
    """Evaluate and compare both models"""
    print("\n" + "="*60)
    print("STEP 3: Model Evaluation")
    print("="*60)
    
    # Full model accuracy
    full_acc = accuracy_score(y_test, full_model.predict(X_test))
    print(f"\nFull Model (60 questions) - Test Accuracy: {full_acc:.4f} ({full_acc*100:.2f}%)")
    
    # Short model accuracy
    X_test_short = X_test[top_features]
    short_acc = accuracy_score(y_test, short_model.predict(X_test_short))
    print(f"Short Model ({TOP_N_FEATURES} questions) - Test Accuracy: {short_acc:.4f} ({short_acc*100:.2f}%)")
    
    # Comparison
    acc_diff = full_acc - short_acc
    acc_retention = (short_acc / full_acc) * 100
    print(f"\nAccuracy Difference: {acc_diff:.4f} ({acc_diff*100:.2f}%)")
    print(f"Accuracy Retention: {acc_retention:.2f}% of full model performance")
    
    return full_acc, short_acc


def convert_to_onnx(model, n_features, output_path):
    """Convert XGBoost model to ONNX format"""
    if not ONNX_AVAILABLE:
        print("ONNX libraries not available, skipping conversion.")
        return False
    
    print(f"\nConverting model to ONNX: {output_path}")
    
    # Register XGBoost converter
    update_registered_converter(
        XGBClassifier, 
        'XGBClassifier',
        calculate_linear_classifier_output_shapes, 
        convert_xgboost,
        options={'nocl': [True, False], 'zipmap': [True, False, 'columns']}
    )
    
    # Define input type
    initial_type = [('float_input', FloatTensorType([None, n_features]))]
    
    # Convert
    onx = convert_sklearn(
        model, 
        initial_types=initial_type,
        options={id(model): {'zipmap': False}}
    )
    
    # Save
    with open(output_path, "wb") as f:
        f.write(onx.SerializeToString())
    
    print(f"ONNX model saved: {output_path}")
    print(f"File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
    
    # Verify the model works
    sess = ort.InferenceSession(output_path)
    print("ONNX model verification: OK")
    
    return True


def save_outputs(short_model, top_features, top_feature_indices, feature_names, le):
    """Save model and metadata for website deployment"""
    print("\n" + "="*60)
    print("STEP 4: Saving Outputs")
    print("="*60)
    
    # Ensure API directory exists
    os.makedirs(API_DIR, exist_ok=True)
    
    # 1. Save top 35 questions metadata
    questions_data = {
        'count': len(top_features),
        'indices': top_feature_indices,  # Indices in original 60-question order
        'features': top_features,  # Feature names
    }
    
    questions_path = os.path.join(API_DIR, 'top_35_questions.json')
    with open(questions_path, 'w') as f:
        json.dump(questions_data, f, indent=2)
    print(f"Saved: {questions_path}")
    
    # 2. Save labels (same as full model)
    labels_path = os.path.join(API_DIR, 'labels.json')
    with open(labels_path, 'w') as f:
        json.dump(list(le.classes_), f)
    print(f"Saved: {labels_path}")
    
    # 3. Convert and save ONNX model
    onnx_path = os.path.join(API_DIR, 'mbti_model_short.onnx')
    convert_to_onnx(short_model, len(top_features), onnx_path)
    
    # 4. Also save joblib as backup
    joblib_path = os.path.join(SCRIPT_DIR, 'xgb_model_short.joblib')
    joblib.dump(short_model, joblib_path)
    print(f"Saved backup: {joblib_path}")
    
    # 5. Save feature ranking for reference
    ranking_path = os.path.join(SCRIPT_DIR, f'top_{TOP_N_FEATURES}_features.csv')
    pd.DataFrame({'Feature': top_features, 'Rank': range(1, len(top_features)+1)}).to_csv(
        ranking_path, index=False
    )
    print(f"Saved: {ranking_path}")


def main():
    print("="*60)
    print("XGBOOST SHORT MODEL TRAINING")
    print(f"Selecting Top {TOP_N_FEATURES} Features")
    print("="*60)
    
    # Load data
    X, y, feature_names = load_and_preprocess_data()
    
    # Split data (exact same as Feature_Ranking_Analysis.ipynb)
    print("\nSplitting data (70% train, 15% val, 15% test)...")
    X_train, X_val, X_test, y_train, y_val, y_test, le = split_data(X, y)
    
    # Train full model and get feature importance
    full_model, importance_df = train_full_model_and_get_importance(
        X_train, X_val, y_train, y_val, feature_names, le
    )
    
    # Get top N features
    top_features = importance_df['Feature'].head(TOP_N_FEATURES).tolist()
    print(f"\nTop {TOP_N_FEATURES} Features Selected:")
    for i, feat in enumerate(top_features, 1):
        print(f"  {i:2d}. {feat[:70]}...")
    
    # Train short model
    short_model, top_feature_indices = train_short_model(
        X_train, X_val, X_test, y_train, y_val, y_test,
        top_features, le, feature_names
    )
    
    # Evaluate and compare
    full_acc, short_acc = evaluate_models(
        full_model, short_model, X_test, y_test, top_features, le
    )
    
    # Save outputs
    save_outputs(short_model, top_features, top_feature_indices, feature_names, le)
    
    # Final summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Full Model (60 questions): {full_acc:.4f} ({full_acc*100:.2f}%)")
    print(f"Short Model ({TOP_N_FEATURES} questions): {short_acc:.4f} ({short_acc*100:.2f}%)")
    print(f"Accuracy Retention: {(short_acc/full_acc)*100:.1f}%")
    print("\nFiles created:")
    print(f"  - {os.path.join(API_DIR, 'mbti_model_short.onnx')}")
    print(f"  - {os.path.join(API_DIR, 'top_35_questions.json')}")
    print("="*60)


if __name__ == "__main__":
    main()
