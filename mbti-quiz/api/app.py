"""
Flask API for MBTI Personality Type Prediction
Uses trained ONNX model to predict personality types from quiz answers
Updated for Vercel deployment
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import numpy as np
import os
import onnxruntime as ort

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

# Load the model and label encoder
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'mbti_model.onnx')
LABELS_PATH = os.path.join(os.path.dirname(__file__), 'labels.json')

ort_session = None
class_labels = None

def load_model():
    global ort_session, class_labels
    try:
        ort_session = ort.InferenceSession(MODEL_PATH)
        with open(LABELS_PATH, 'r') as f:
            class_labels = json.load(f)
        print("Model and labels loaded successfully!")
        print(f"Classes: {class_labels}")
    except Exception as e:
        print(f"Error loading model: {e}")
        print(f"Expected model at: {MODEL_PATH}")
        print(f"Expected labels at: {LABELS_PATH}")

# Load model immediately when module is imported
load_model()


@app.route('/health', methods=['GET'])
@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': ort_session is not None
    })


@app.route('/predict', methods=['POST'])
@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict MBTI personality type from quiz answers"""
    if ort_session is None or class_labels is None:
        return jsonify({
            'error': 'Model not loaded.'
        }), 500

    try:
        data = request.get_json()
        answers = data.get('answers', [])

        if len(answers) != 60:
            return jsonify({
                'error': f'Expected 60 answers, got {len(answers)}'
            }), 400

        # Convert answers to numpy array (float32 required for ONNX)
        X = np.array(answers, dtype=np.float32).reshape(1, -1)

        # Get input name
        input_name = ort_session.get_inputs()[0].name
        
        # Run inference
        outputs = ort_session.run(None, {input_name: X})
        
        # Output format depends on ONNX conversion
        # Usually standard classifiers return [label, probabilities] 
        # But for XGBoost via onnxmltools, it might return [class_probs] or similar
        # Based on 'multi:softprob' training, it likely returns probabilities directly as the second output 
        # or a single output of probabilities if configured that way.
        # Let's inspect outputs: onnxmltools usually produces: [label_tensor, probability_tensor_map]
        
        # Inspecting the typical conversion:
        # returns (label, probabilities)
        
        # XGBoost ONNX output usually:
        # 0: Label (int64)
        # 1: Probabilities (sequence of map) or tensor
        
        # Let's handle the output safely
        if len(outputs) >= 2:
             probabilities = outputs[1][0] # Map or list
             # If it's a list of maps (one per sample)
             if isinstance(probabilities, dict):
                 # Convert map to list based on class_labels order
                 # onnxruntime returns map {class_index: prob} or {class_label: prob}
                 # Since we didn't store labels in the model (we use generic labels or indices), 
                 # let's assume keys are indices matching our class_labels
                 prob_list = [probabilities[i] for i in range(len(class_labels))]
                 prediction_idx = np.argmax(prob_list)
             else:
                 # It might be a tensor of shape (1, 16)
                 prob_list = probabilities
                 prediction_idx = np.argmax(prob_list)
        else:
             # Assume single output is probabilities
             prob_list = outputs[0][0]
             prediction_idx = np.argmax(prob_list)

        # Decode predicted class
        predicted_type = class_labels[int(prediction_idx)]

        # Create probabilities dictionary
        prob_dict = {
            class_labels[i]: float(prob_list[i])
            for i in range(len(class_labels))
        }

        # Get confidence (max probability)
        confidence = float(np.max(prob_list))

        return jsonify({
            'predicted_type': predicted_type,
            'confidence': confidence,
            'probabilities': prob_dict
        })

    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500


@app.route('/types', methods=['GET'])
@app.route('/api/types', methods=['GET'])
def get_types():
    """Get all possible personality types"""
    if class_labels is None:
        # Fallback if not loaded, though load_model() is called at module level
        return jsonify({
            'types': ['INTJ', 'INTP', 'ENTJ', 'ENTP', 'INFJ', 'INFP', 'ENFJ', 'ENFP',
                     'ISTJ', 'ISFJ', 'ESTJ', 'ESFJ', 'ISTP', 'ISFP', 'ESTP', 'ESFP']
        })
    
    return jsonify({
        'types': class_labels
    })


if __name__ == '__main__':
    # load_model() is already called at module level
    print("\n" + "="*50)
    print("MBTI Prediction API")
    print("="*50)
    print("Endpoints:")
    print("  GET  /health  - Health check")
    print("  POST /predict - Predict personality type")
    print("  GET  /types   - Get all personality types")
    print("="*50 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=True)
