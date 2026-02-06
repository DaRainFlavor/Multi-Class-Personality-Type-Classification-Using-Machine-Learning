
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import os

# Set Plot Style
sns.set(style="whitegrid")

def main():
    print("="*50)
    print("FEATURE SELECTION ANALYSIS USING XGBOOST")
    print("="*50)

    # 1. Load Data
    # Assuming the script is in Feature_Selection_Analysis/ and data is in root or Initial works/
    # We will try to find the file dynamically or use a relative path
    data_path = '../Initial works/16P_cleaned.csv' 
    if not os.path.exists(data_path):
        data_path = '../16P.csv' # Fallback
    
    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at {data_path}")
        return

    print(f"[1] Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"    Shape: {df.shape}")

    # 2. Preprocessing
    print("\n[2] Preprocessing...")
    # Drop identifier if exists
    if 'Response Id' in df.columns:
        df = df.drop(columns=['Response Id'])
    
    # Identify Question Columns (features)
    # The dataset has many columns, we only want the 60 questions
    # The cleared dataset might have other columns like 'E_I_score', etc.
    # We need to filter for just the questions. 
    # Usually questions are the first 60 columns after ID if cleaned properly, 
    # OR we explicitly exclude known derivative columns.
    
    # Let's drop known target/derivative columns to be safe
    cols_to_drop = ['Personality', 'E_I_score', 'S_N_score', 'T_F_score', 'J_P_score', 
                    'E_I_strength', 'S_N_strength', 'T_F_strength', 'J_P_strength',
                    'is_Extraverted', 'is_Intuitive', 'is_Feeling', 'is_Judging', 
                    'Consistency', 'Original_Personality']
    
    # Filter columns that actually exist in the dataframe
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    
    X = df.drop(columns=cols_to_drop)
    y = df['Personality']
    
    feature_names = X.columns.tolist()
    print(f"    Features: {len(feature_names)} questions")
    print(f"    Target: Personality ({y.nunique()} classes)")

    # Encode Target
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Split Data (70% Train, 15% Validation, 15% Test)
    # 1. Split into Train+Val (85%) and Test (15%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y_encoded, 
        test_size=0.15, 
        random_state=42, 
        stratify=y_encoded
    )
    
    # 2. Split Train+Val into Train (70% of total) and Val (15% of total)
    # 0.15 / 0.85 = 0.17647...
    val_size_adjusted = 0.15 / 0.85
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, 
        test_size=val_size_adjusted, 
        random_state=42, 
        stratify=y_temp
    )
    
    print(f"    Train: {len(X_train)} (70%), Val: {len(X_val)} (15%), Test: {len(X_test)} (15%)")
    
    # 3. XGBoost Feature Importance
    print("\n[3] Training XGBoost for Feature Importance...")
    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        n_jobs=-1,
        objective='multi:softmax',
        num_class=len(le.classes_)
    )
    model.fit(X_train, y_train)
    
    baseline_acc = accuracy_score(y_test, model.predict(X_test))
    print(f"    Baseline Accuracy (All 60 features): {baseline_acc:.4f}")

    # Get Importance
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False).reset_index(drop=True)
    
    # Save Ranking
    feature_importance_df.to_csv('feature_ranking.csv', index=False)
    print("    Saved ranking to 'feature_ranking.csv'")

    # 4. Plot Feature Importance (Top 20)
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(20), palette='viridis')
    plt.title('Top 20 Most Important Questions')
    plt.xlabel('Importance Score')
    plt.ylabel('Question')
    plt.tight_layout()
    plt.savefig('top_20_features.png')
    print("    Saved plot 'top_20_features.png'")

    # 5. Recursive Analysis (Stepwise Top N)
    print("\n[4] Performing Top-N Feature Analysis...")
    # 5. Recursive Analysis (Stepwise Top N)
    print("\n[4] Performing Top-N Feature Analysis (Fine-Grained)...")
    # Check Top 1-20 individually to see Pareto effect, then every 5
    n_features_list = list(range(1, 21)) + list(range(25, 65, 5))
    results = []

    for n in n_features_list:
        top_n_feats = feature_importance_df['Feature'].head(n).tolist()
        
        # Subset data
        X_train_sub = X_train[top_n_feats]
        X_test_sub = X_test[top_n_feats]
        
        # Train new model
        model_sub = XGBClassifier(
            n_estimators=100, 
            learning_rate=0.1, 
            max_depth=5, 
            random_state=42, 
            n_jobs=-1,
            validate_parameters=False, # Speed up
            objective='multi:softmax',
            num_class=len(le.classes_)    
        )
        model_sub.fit(X_train_sub, y_train)
        acc = accuracy_score(y_test, model_sub.predict(X_test_sub))
        
        results.append({'N_Features': n, 'Accuracy': acc})
        print(f"    Top {n}: Acc = {acc:.4f}")

    results_df = pd.DataFrame(results)
    results_df.to_csv('top_n_accuracy.csv', index=False)

    # 6. Plot Accuracy vs N Features
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='N_Features', y='Accuracy', data=results_df, marker='o', linewidth=2.5)
    plt.axhline(y=baseline_acc, color='r', linestyle='--', label=f'Baseline (60): {baseline_acc:.4f}')
    plt.title('Accuracy vs Number of Top Features')
    plt.xlabel('Number of Questions (Ranked by Importance)')
    plt.ylabel('Test Accuracy')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('accuracy_vs_n_features.png')
    print("    Saved plot 'accuracy_vs_n_features.png'")
    
    # 7. Recommendation
    # Simple elbow detection or threshold (e.g., 95% of baseline)
    threshold = 0.98 * baseline_acc
    rec_n = results_df[results_df['Accuracy'] >= threshold]['N_Features'].min()
    
    print("\n" + "="*50)
    print("RECOMMENDATION")
    print("="*50)
    print(f"To maintain at least 98% of the baseline accuracy ({threshold:.4f}):")
    print(f"Keep the Top {rec_n} questions.")
    
    print("\nAnalysis Complete.")

if __name__ == "__main__":
    main()
