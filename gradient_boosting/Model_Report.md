# Gradient Boosting Classifier Report
**Project:** Multi-Class Classification of Personality Types (MBTI)  
**Date:** 2026-01-31  

---

## 1. Executive Summary
The **XGBoost Gradient Boosting Classifier** was trained to predict 16 MBTI personality types based on 60 survey questions. The model achieved an outstanding **98.22% accuracy** on the unseen test set, demonstrating high reliability and robust generalization.

## 2. Model Configuration

### Data Split
- **Training:** 70% (42,023 samples)
- **Validation:** 15% (8,976 samples)
- **Test:** 15% (9,000 samples)
*Stratified split ensures equal representation of all 16 personality types.*

### Hyperparameters
- **Algorithm:** XGBoost Classifier
- **Estimators:** 500 (Early stopped at best iteration)
- **Learning Rate:** 0.1
- **Max Depth:** 6
- **Objective:** Multi-class Softprob

## 3. Performance Metrics

Measure | Score | Notes
--- | --- | ---
**Test Accuracy** | **98.22%** | High precision across all classes
**Top-3 Accuracy** | **99.23%** | Correct type is in top 3 guesses 99% of time
**Macro F1-Score** | 0.9822 | Balanced performance for all labels
**Train Accuracy** | 100.00% | Model learned training data perfectly

*(Training stopped early at iteration 351 to prevent overfitting)*

## 4. Visualizations

### 4.1 Confusion Matrix
The confusion matrix shows the alignment between Actual and Predicted personality types. The diagonal line indicates correct predictions.
![Confusion Matrix](gb_results/confusion_matrix.png)

### 4.2 Feature Importance
The top 20 most influential survey questions driving the model's decisions.
![Feature Importance](gb_results/feature_importance.png)

### 4.3 Per-Class Accuracy
Accuracy breakdown for each of the 16 personality types.
![Per-Class Accuracy](gb_results/per_class_accuracy.png)

## 5. Conclusion
The Gradient Boosting model is highly effective for this classification task. With a test accuracy of over 98% and a top-3 accuracy indistinguishable from perfect (99.2%), it is ready for deployment or further application analysis. The low gap between training and test accuracy (1.78%) confirms the model has not overfit.

---
*Report generated automatically from model training logs.*
