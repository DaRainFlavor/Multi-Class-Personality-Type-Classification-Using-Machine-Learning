"""
Flask API for MBTI Personality Type Prediction
Supports both FULL (60 questions) and SHORT (35 questions) models
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

# File paths
API_DIR = os.path.dirname(__file__)
FULL_MODEL_PATH = os.path.join(API_DIR, 'mbti_model.onnx')
SHORT_MODEL_PATH = os.path.join(API_DIR, 'mbti_model_short.onnx')
LABELS_PATH = os.path.join(API_DIR, 'labels.json')
TOP_35_PATH = os.path.join(API_DIR, 'top_35_questions.json')

# Global variables for loaded models
full_session = None
short_session = None
class_labels = None
top_35_indices = None
init_error = None


def load_models():
    """Load both full and short models"""
    global full_session, short_session, class_labels, top_35_indices, init_error
    
    try:
        # Load full model (60 questions)
        if os.path.exists(FULL_MODEL_PATH):
            full_session = ort.InferenceSession(FULL_MODEL_PATH)
            print(f"✓ Full model loaded: {FULL_MODEL_PATH}")
        else:
            print(f"✗ Full model not found: {FULL_MODEL_PATH}")
        
        # Load short model (35 questions)
        if os.path.exists(SHORT_MODEL_PATH):
            short_session = ort.InferenceSession(SHORT_MODEL_PATH)
            print(f"✓ Short model loaded: {SHORT_MODEL_PATH}")
        else:
            print(f"✗ Short model not found: {SHORT_MODEL_PATH}")
        
        # Load labels
        if os.path.exists(LABELS_PATH):
            with open(LABELS_PATH, 'r') as f:
                class_labels = json.load(f)
            print(f"✓ Labels loaded: {len(class_labels)} classes")
        
        # Load top 35 question indices
        if os.path.exists(TOP_35_PATH):
            with open(TOP_35_PATH, 'r') as f:
                top_35_data = json.load(f)
                top_35_indices = top_35_data.get('indices', [])
            print(f"✓ Top 35 indices loaded: {len(top_35_indices)} questions")
        
        print("Models loaded successfully!")
        
    except Exception as e:
        print(f"Error loading models: {e}")
        init_error = str(e)


# Load models at startup
load_models()


@app.route('/health', methods=['GET'])
@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'full_model_loaded': full_session is not None,
        'short_model_loaded': short_session is not None,
        'labels_loaded': class_labels is not None
    })


@app.route('/predict', methods=['POST'])
@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict MBTI personality type from quiz answers
    
    Accepts:
    - 60 answers: Uses full model
    - 35 answers: Uses short model (must be in correct feature order)
    - {"answers": [...], "mode": "short"}: Uses short model with index mapping
    """
    try:
        data = request.get_json()
        answers = data.get('answers', [])
        mode = data.get('mode', 'auto')  # 'full', 'short', or 'auto'
        
        # Determine which model to use
        if mode == 'short' or len(answers) == 35:
            # Short model (35 questions)
            if short_session is None:
                return jsonify({'error': 'Short model not loaded'}), 500
            
            if len(answers) != 35:
                return jsonify({'error': f'Short mode requires 35 answers, got {len(answers)}'}), 400
            
            session = short_session
            X = np.array(answers, dtype=np.float32).reshape(1, -1)
            model_used = 'short'
            
        elif mode == 'full' or len(answers) == 60:
            # Full model (60 questions)
            if full_session is None:
                return jsonify({'error': 'Full model not loaded'}), 500
            
            if len(answers) != 60:
                return jsonify({'error': f'Full mode requires 60 answers, got {len(answers)}'}), 400
            
            session = full_session
            X = np.array(answers, dtype=np.float32).reshape(1, -1)
            model_used = 'full'
            
        else:
            return jsonify({
                'error': f'Invalid number of answers: {len(answers)}. Expected 60 (full) or 35 (short).'
            }), 400
        
        # Run inference
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: X})
        
        # Process outputs
        if len(outputs) >= 2:
            probabilities = outputs[1][0]
            if isinstance(probabilities, dict):
                prob_list = [probabilities[i] for i in range(len(class_labels))]
            else:
                prob_list = probabilities
        else:
            prob_list = outputs[0][0]
        
        prediction_idx = int(np.argmax(prob_list))
        predicted_type = class_labels[prediction_idx]
        confidence = float(np.max(prob_list))
        
        # Create probabilities dictionary
        prob_dict = {
            class_labels[i]: float(prob_list[i])
            for i in range(len(class_labels))
        }
        
        return jsonify({
            'predicted_type': predicted_type,
            'confidence': confidence,
            'probabilities': prob_dict,
            'model_used': model_used,
            'questions_answered': len(answers)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/questions/short', methods=['GET'])
@app.route('/api/questions/short', methods=['GET'])
def get_short_questions():
    """Get the indices and details of the 35 short questions"""
    if top_35_indices is None:
        return jsonify({'error': 'Top 35 questions not loaded'}), 500
    
    try:
        with open(TOP_35_PATH, 'r') as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/types', methods=['GET'])
@app.route('/api/types', methods=['GET'])
def get_types():
    """Get all possible personality types"""
    if class_labels is None:
        return jsonify({
            'types': ['INTJ', 'INTP', 'ENTJ', 'ENTP', 'INFJ', 'INFP', 'ENFJ', 'ENFP',
                     'ISTJ', 'ISFJ', 'ESTJ', 'ESFJ', 'ISTP', 'ISFP', 'ESTP', 'ESFP']
        })
    return jsonify({'types': class_labels})


if __name__ == '__main__':
    print("\n" + "="*50)
    print("MBTI Prediction API")
    print("="*50)
    print("Endpoints:")
    print("  GET  /health          - Health check")
    print("  POST /predict         - Predict personality (60 or 35 questions)")
    print("  GET  /questions/short - Get short questionnaire details")
    print("  GET  /types           - Get all personality types")
    print("="*50 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=True)
