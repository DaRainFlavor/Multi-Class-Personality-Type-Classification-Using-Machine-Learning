import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# 1. Load Data
df = pd.read_csv('16P_cleaned_simple.csv', encoding='cp1252')

# 2. Preprocessing
# Drop the ID column as it's not useful for prediction
df = df.drop(columns=['Response Id'])

# Separate Features (X) and Target (y)
X = df.drop(columns=['Personality'])
y = df['Personality']

# 3. Check for non-numeric values in features and convert them
print("Checking data types and converting non-numeric values...")
for col in X.columns:
    # Check if column contains non-numeric values
    if X[col].dtype == 'object':
        print(f"Converting non-numeric column: {col}")
        # Use LabelEncoder to convert string categories to numbers
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

# Convert all feature columns to numeric, coercing errors to NaN
X = X.apply(pd.to_numeric, errors='coerce')

# Check for any remaining NaN values and fill them
if X.isnull().any().any():
    print("Warning: NaN values found in features. Filling with column means.")
    X = X.fillna(X.mean())

# 4. Encode the target variable if it's categorical
if y.dtype == 'object':
    print("Encoding target variable...")
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

# 5. Split Data into Training and Testing Sets (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
print("Training model...")
model.fit(X_train, y_train)

# 7. Make Predictions
y_pred = model.predict(X_test)

# 8. Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")

# If we encoded the target, show original labels in classification report
if 'label_encoder' in locals():
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
else:
    print(classification_report(y_test, y_pred))

# 9. Read answers from file and make prediction
print("\n--- Predicting Personality from answers.txt ---")

def read_answers_from_file(filename):
    """Read answers from text file and return as dictionary"""
    answers = {}
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if line and ':' in line:
                    # Split question and answer
                    question, answer = line.split(':', 1)
                    question = question.strip()
                    answer = answer.strip()
                    
                    # Convert answer to float
                    try:
                        answers[question] = float(answer)
                    except ValueError:
                        print(f"Warning: Could not convert answer '{answer}' to number for question: {question}")
                        answers[question] = 0.0  # Default value if conversion fails
        return answers
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

# Read answers from file
answers = read_answers_from_file('answers3.txt')

if answers:
    print(f"Successfully read {len(answers)} answers from file.")
    
    # Create DataFrame with the same column order as training data
    my_data = {}
    missing_questions = 0
    
    for col in X.columns:
        # Try to find matching question (case-insensitive partial match)
        found = False
        for question in answers.keys():
            if col.lower() in question.lower() or question.lower() in col.lower():
                my_data[col] = [answers[question]]
                found = True
                break
        
        if not found:
            print(f"Warning: No matching answer found for feature: '{col}'. Using default value 0.")
            my_data[col] = [0.0]
            missing_questions += 1
    
    if missing_questions > 0:
        print(f"Total missing answers: {missing_questions}")
    
    # Create DataFrame and predict
    my_df = pd.DataFrame(my_data)
    
    # Ensure the column order matches training data
    my_df = my_df[X.columns]
    
    # Predict
    my_prediction_encoded = model.predict(my_df)
    
    # Decode the prediction if we encoded the target
    if 'label_encoder' in locals():
        my_prediction = label_encoder.inverse_transform(my_prediction_encoded)
        print(f"\nBased on the answers from answers.txt, the predicted personality is: {my_prediction[0]}")
    else:
        print(f"\nBased on the answers from answers.txt, the predicted personality is: {my_prediction_encoded[0]}")
    
    # Show probability distribution
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(my_df)
        if 'label_encoder' in locals():
            personality_types = label_encoder.classes_
        else:
            personality_types = model.classes_
        
        print("\nProbability distribution across personality types:")
        # Create list of (personality, probability) pairs and sort by probability
        prob_pairs = list(zip(personality_types, probabilities[0]))
        prob_pairs.sort(key=lambda x: x[1], reverse=True)
        
        for personality, prob in prob_pairs:
            print(f"  {personality}: {prob*100:.2f}%")
else:
    print("Failed to read answers from file.")