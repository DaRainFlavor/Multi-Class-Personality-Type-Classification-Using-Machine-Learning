"""
=============================================================================
CELLS TO ADD TO Feature_Ranking_Analysis.ipynb FOR MODEL EXPORT
=============================================================================
Copy each "# === CELL ===" section as a NEW CELL in your Colab notebook.
Add these AFTER the "5. Recommendation" section.

Files will be auto-downloaded after running.
=============================================================================
"""

# === CELL 1: Install ONNX Libraries ===
# Add as new MARKDOWN cell:
"""
## 6. Export Models for Website Deployment
This section trains both the **full model (60 questions)** and **short model (35 questions)**, 
converts them to ONNX format, and auto-downloads the files needed for website deployment.
"""

# Add as new CODE cell:
"""
# Install ONNX conversion libraries
!pip install onnx onnxruntime skl2onnx onnxmltools
"""


# === CELL 2: ONNX Converter Function ===
# Add as new CODE cell:
"""
# Import ONNX libraries
from skl2onnx import convert_sklearn, update_registered_converter
from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes
from onnxmltools.convert.xgboost.operator_converters.XGBClassifier import convert_xgboost
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as ort
import json

def convert_to_onnx(model, n_features, output_path):
    '''Convert XGBoost model to ONNX format'''
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
    with open(output_path, 'wb') as f:
        f.write(onx.SerializeToString())
    
    # Verify
    sess = ort.InferenceSession(output_path)
    print(f'✅ Saved & verified: {output_path}')
    return True

print('ONNX conversion function ready!')
"""


# === CELL 3: Train Short Model ===
# Add as new CODE cell:
"""
# === TRAIN SHORT MODEL (35 Questions) ===
TOP_N = 35
top_35_features = feature_importance_df['Feature'].head(TOP_N).tolist()

# Get indices in original order (for mapping answers)
top_35_indices = [feature_names.index(feat) for feat in top_35_features]

# Subset data
X_train_short = X_train[top_35_features]
X_val_short = X_val[top_35_features]
X_test_short = X_test[top_35_features]

# Train short model
print(f'Training Short Model ({TOP_N} questions)...')
model_short = XGBClassifier(**xgb_params, early_stopping_rounds=15)
model_short.fit(X_train_short, y_train, eval_set=[(X_val_short, y_val)], verbose=False)

short_acc = accuracy_score(y_test, model_short.predict(X_test_short))
print(f'Short Model Accuracy: {short_acc:.4f} ({short_acc*100:.2f}%)')
print(f'Accuracy Retention: {(short_acc/baseline_acc)*100:.1f}% of full model')
"""


# === CELL 4: Export All Files ===
# Add as new CODE cell:
"""
# === EXPORT ALL FILES ===
print('='*50)
print('EXPORTING FILES FOR WEBSITE DEPLOYMENT')
print('='*50)

# 1. Full Model ONNX (already trained as 'model')
convert_to_onnx(model, n_features=60, output_path='mbti_model.onnx')

# 2. Short Model ONNX
convert_to_onnx(model_short, n_features=TOP_N, output_path='mbti_model_short.onnx')

# 3. Labels (class names)
labels = list(le.classes_)
with open('labels.json', 'w') as f:
    json.dump(labels, f)
print('✅ Saved: labels.json')

# 4. Top 35 Questions metadata
top_35_data = {
    'count': TOP_N,
    'indices': top_35_indices,
    'features': top_35_features
}
with open('top_35_questions.json', 'w') as f:
    json.dump(top_35_data, f, indent=2)
print('✅ Saved: top_35_questions.json')

# 5. All questions list (for reference)
with open('all_questions.json', 'w') as f:
    json.dump(feature_names, f, indent=2)
print('✅ Saved: all_questions.json')

# 6. Accuracy report
report = {
    'full_model_accuracy': round(baseline_acc, 4),
    'short_model_accuracy': round(short_acc, 4),
    'accuracy_retention_percent': round((short_acc/baseline_acc)*100, 2),
    'full_questions': 60,
    'short_questions': TOP_N
}
with open('accuracy_report.json', 'w') as f:
    json.dump(report, f, indent=2)
print('✅ Saved: accuracy_report.json')

print('\\n' + '='*50)
print('SUMMARY')
print('='*50)
print(f'Full Model (60Q): {baseline_acc*100:.2f}%')
print(f'Short Model ({TOP_N}Q): {short_acc*100:.2f}%')
print(f'Accuracy Retention: {(short_acc/baseline_acc)*100:.1f}%')
"""


# === CELL 5: Auto-Download Files ===
# Add as new CODE cell:
"""
# === AUTO-DOWNLOAD FILES (Colab) ===
from google.colab import files

print('Downloading files...')
print('(Click "Allow" if prompted)')

# Download all generated files
files.download('mbti_model.onnx')         # Full model
files.download('mbti_model_short.onnx')   # Short model
files.download('labels.json')             # Class labels
files.download('top_35_questions.json')   # Short model question indices
files.download('all_questions.json')      # All 60 questions
files.download('accuracy_report.json')    # Accuracy comparison

print('\\n✅ All files downloaded!')
"""


# =============================================================================
# WHERE TO PLACE DOWNLOADED FILES IN YOUR REPO:
# =============================================================================
#
# | File                    | Location                         |
# |-------------------------|----------------------------------|
# | mbti_model.onnx         | mbti-quiz/api/                   |
# | mbti_model_short.onnx   | mbti-quiz/api/                   |
# | labels.json             | mbti-quiz/api/                   |
# | top_35_questions.json   | mbti-quiz/api/                   |
# | all_questions.json      | mbti-quiz/api/ (optional)        |
# | accuracy_report.json    | Feature_Selection_Analysis/      |
#
# =============================================================================
