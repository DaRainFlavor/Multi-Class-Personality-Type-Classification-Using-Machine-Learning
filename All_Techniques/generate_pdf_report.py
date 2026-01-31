from fpdf import FPDF
import os

print("Starting PDF generation for Model Comparison Report...")

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'Model Comparison Report', 0, 1, 'C')
        self.set_font('Arial', '', 10)
        self.cell(0, 8, 'Multi-Class Classification of Personality Types (MBTI)', 0, 1, 'C')
        self.set_font('Arial', 'I', 9)
        self.cell(0, 6, 'Comparing 4 Machine Learning Techniques', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, 'Page ' + str(self.page_no()) + '/{nb}', 0, 0, 'C')

    def chapter_title(self, title, level=1):
        self.set_font('Arial', 'B', 14 if level == 1 else 11)
        # Rainbow colors for different sections
        colors = [(100, 149, 237), (60, 179, 113), (255, 165, 0), (147, 112, 219)]
        color = colors[(self.page_no() - 1) % len(colors)]
        self.set_fill_color(*color)
        self.cell(0, 8, title, 0, 1, 'L', level == 1)
        self.ln(2)

    def add_table(self, headers, data, col_widths):
        # Header
        self.set_font('Arial', 'B', 9)
        self.set_fill_color(200, 220, 255)
        for i, header in enumerate(headers):
            self.cell(col_widths[i], 7, header, 1, 0, 'C', True)
        self.ln()
        
        # Data
        self.set_font('Arial', '', 8)
        for row in data:
            for i, cell in enumerate(row):
                self.cell(col_widths[i], 6, str(cell), 1, 0, 'C')
            self.ln()
        self.ln(3)

def create_pdf():
    pdf = PDF()
    pdf.alias_nb_pages()
    pdf.add_page()
    
    # Executive Summary
    pdf.chapter_title('Executive Summary')
    pdf.set_font('Arial', '', 10)
    pdf.multi_cell(0, 5,
        "This report compares four machine learning algorithms trained to classify 16 MBTI personality types. "
        "All models used identical data splits (70/15/15) with random state 42 for fair comparison."
    )
    pdf.ln(3)
    
    # Quick Results
    pdf.set_font('Arial', 'B', 10)
    pdf.cell(0, 6, "Quick Results Overview:", 0, 1)
    pdf.set_font('Arial', '', 9)
    
    headers = ['Rank', 'Model', 'Test Acc', 'Top-3 Acc']
    data = [
        ['1st', 'XGBoost (Gradient Boosting)', '98.22%', '99.23%'],
        ['2nd', 'Random Forest', '97.57%', '99.08%'],
        ['3rd', 'Logistic Regression', '91.90%', '98.62%'],
        ['4th', 'Linear Discriminant Analysis', '90.56%', '98.14%']
    ]
    pdf.add_table(headers, data, [15, 80, 30, 30])
    
    # Dataset Configuration
    pdf.chapter_title('1. Dataset Configuration')
    pdf.set_font('Arial', '', 9)
    pdf.multi_cell(0, 5,
        "Total Samples: 59,999\n"
        "Training Set: 42,023 samples (70%)\n"
        "Validation Set: 8,976 samples (15%)\n"
        "Test Set: 9,000 samples (15%)\n"
        "Features: 60 survey questions\n"
        "Target Classes: 16 MBTI personality types"
    )
    pdf.ln(5)
    
    # Performance Comparison
    pdf.chapter_title('2. Accuracy Metrics Comparison')
    headers = ['Metric', 'XGBoost', 'Rand Forest', 'Log Reg', 'LDA']
    data = [
        ['Test Accuracy', '98.22%', '97.57%', '91.90%', '90.56%'],
        ['Validation Acc', '98.12%', '97.48%', '92.12%', '90.63%'],
        ['Train Accuracy', '100.00%', '99.34%', '92.35%', '90.53%'],
        ['Train-Test Gap', '1.78%', '1.77%', '0.45%', '0.00%']
    ]
    pdf.add_table(headers, data, [40, 30, 30, 25, 25])
    
    # Top-K Accuracy
    pdf.chapter_title('3. Top-K Accuracy (Test Set)')
    headers = ['K', 'XGBoost', 'Rand Forest', 'Log Reg', 'LDA']
    data = [
        ['Top-1', '98.22%', '97.57%', '91.90%', '90.56%'],
        ['Top-2', '99.10%', '98.77%', '97.31%', '96.44%'],
        ['Top-3', '99.23%', '99.08%', '98.62%', '98.14%'],
        ['Top-5', '99.38%', '99.28%', '99.27%', '99.01%']
    ]
    pdf.add_table(headers, data, [20, 35, 35, 30, 30])
    
    # Aggregate Metrics
    pdf.set_font('Arial', 'B', 10)
    pdf.cell(0, 6, "Aggregate Performance Metrics:", 0, 1)
    pdf.ln(2)
    headers = ['Metric', 'XGBoost', 'Rand Forest', 'Log Reg', 'LDA']
    data = [
        ['Macro F1', '0.9822', '0.9757', '0.9189', '0.9053'],
        ['Macro Prec', '0.9823', '0.9757', '0.9190', '0.9055'],
        ['Macro Recall', '0.9822', '0.9757', '0.9190', '0.9055']
    ]
    pdf.add_table(headers, data, [40, 30, 30, 25, 25])
    
    # Model Analysis - new page
    pdf.add_page()
    pdf.chapter_title('4. Model-Specific Analysis')
    
    # XGBoost
    pdf.set_font('Arial', 'B', 10)
    pdf.set_fill_color(255, 215, 0)
    pdf.cell(0, 6, '4.1 Gradient Boosting (XGBoost) - WINNER', 0, 1, 'L', True)
    pdf.ln(1)
    pdf.set_font('Arial', '', 9)
    pdf.multi_cell(0, 5,
        "Strengths:\n"
        "  - Highest accuracy (98.22%)\n"
        "  - Superior at handling complex patterns\n"
        "  - Early stopping prevents overfitting\n\n"
        "Weaknesses:\n"
        "  - Longer training time\n"
        "  - More hyperparameters to tune\n"
        "  - Shows some overfitting (100% train accuracy)\n\n"
        "Best For: Maximum prediction accuracy, production deployments"
    )
    pdf.ln(4)
    
    # Random Forest
    pdf.set_font('Arial', 'B', 10)
    pdf.set_fill_color(192, 192, 192)
    pdf.cell(0, 6, '4.2 Random Forest - 2nd Place', 0, 1, 'L', True)
    pdf.ln(1)
    pdf.set_font('Arial', '', 9)
    pdf.multi_cell(0, 5,
        "Strengths:\n"
        "  - Excellent accuracy (97.57%)\n"
        "  - Fast training (100 trees, parallel)\n"
        "  - Good generalization (1.77% gap)\n"
        "  - Clear feature importance\n\n"
        "Weaknesses:\n"
        "  - Slightly lower accuracy than XGBoost\n"
        "  - Larger model size\n\n"
        "Best For: Balance of accuracy and training speed"
    )
    pdf.ln(4)
    
    # Logistic Regression
    pdf.set_font('Arial', 'B', 10)
    pdf.set_fill_color(205, 127, 50)
    pdf.cell(0, 6, '4.3 Logistic Regression - 3rd Place', 0, 1, 'L', True)
    pdf.ln(1)
    pdf.set_font('Arial', '', 9)
    pdf.multi_cell(0, 5,
        "Strengths:\n"
        "  - Very fast training (24 iterations)\n"
        "  - Excellent generalization (0.45% gap)\n"
        "  - Highly interpretable coefficients\n"
        "  - Simple, minimal tuning\n\n"
        "Weaknesses:\n"
        "  - Lower raw accuracy (91.90%)\n"
        "  - Linear boundaries miss complex patterns\n\n"
        "Best For: Fast prototyping, interpretability"
    )
    pdf.ln(4)
    
    # LDA
    pdf.set_font('Arial', 'B', 10)
    pdf.cell(0, 6, '4.4 Linear Discriminant Analysis', 0, 1)
    pdf.ln(1)
    pdf.set_font('Arial', '', 9)
    pdf.multi_cell(0, 5,
        "Strengths:\n"
        "  - Very fast training\n"
        "  - Perfect generalization (no overfitting)\n"
        "  - Dimensionality reduction (15 components)\n"
        "  - No hyperparameter tuning\n\n"
        "Weaknesses:\n"
        "  - Lowest accuracy (90.56%)\n"
        "  - Assumes Gaussian distributions\n"
        "  - Linear decision boundaries\n\n"
        "Best For: Baseline model, dimensionality reduction"
    )
    
    # Challenging Personality Types
    pdf.add_page()
    pdf.chapter_title('5. Most Challenging Personality Types')
    pdf.set_font('Arial', '', 9)
    pdf.multi_cell(0, 5,
        "Some personality types are harder to classify across all models. "
        "ESFJ and ISFJ types show lower F1-scores, especially with linear models."
    )
    pdf.ln(3)
    
    headers = ['Type', 'XGBoost F1', 'RF F1', 'LR F1', 'LDA F1']
    data = [
        ['ESFJ', '0.9813', '0.9747', '0.8670', '0.8239'],
        ['ISFJ', '0.9813', '0.9766', '0.8923', '0.8639'],
        ['ISFP', '0.9751', '0.9596', '0.8961', '0.8863'],
        ['INFJ', '0.9840', '0.9707', '0.9111', '0.9024']
    ]
    pdf.add_table(headers, data, [35, 30, 25, 25, 25])
    
    # Feature Importance
    pdf.chapter_title('6. Top Features (Consensus)')
    pdf.set_font('Arial', '', 9)
    pdf.multi_cell(0, 5,
        "All models identify similar key features:\n\n"
        "1. 'You often end up doing things at the last possible moment'\n"
        "   (Judging vs Perceiving)\n\n"
        "2. 'You are not too interested in discussing interpretations of creative works'\n"
        "   (Intuition vs Sensing)\n\n"
        "3. 'Your happiness comes more from helping others'\n"
        "   (Thinking vs Feeling)\n\n"
        "4. 'You enjoy going to art museums'\n"
        "   (Intuition vs Sensing, Openness)\n\n"
        "5. 'You like to have a to-do list for each day'\n"
        "   (Judging vs Perceiving)"
    )
    pdf.ln(5)
    
    # Recommendations
    pdf.add_page()
    pdf.chapter_title('7. Recommendations')
    
    pdf.set_font('Arial', 'B', 10)
    pdf.set_fill_color(144, 238, 144)
    pdf.cell(0, 6, 'For Maximum Accuracy: Use XGBoost', 0, 1, 'L', True)
    pdf.set_font('Arial', '', 9)
    pdf.multi_cell(0, 5, "Best overall performance (98.22%). Worth the extra complexity for production systems.\n")
    
    pdf.set_font('Arial', 'B', 10)
    pdf.set_fill_color(173, 216, 230)
    pdf.cell(0, 6, 'For Speed & Accuracy Balance: Use Random Forest', 0, 1, 'L', True)
    pdf.set_font('Arial', '', 9)
    pdf.multi_cell(0, 5, "Excellent accuracy (97.57%) with faster training. Great for most applications.\n")
    
    pdf.set_font('Arial', 'B', 10)
    pdf.set_fill_color(255, 228, 181)
    pdf.cell(0, 6, 'For Interpretability: Use Logistic Regression', 0, 1, 'L', True)
    pdf.set_font('Arial', '', 9)
    pdf.multi_cell(0, 5, "Fast training, clear interpretation, good generalization (91.90%).\n")
    
    pdf.set_font('Arial', 'B', 10)
    pdf.set_fill_color(221, 160, 221)
    pdf.cell(0, 6, 'For Baseline: Use LDA', 0, 1, 'L', True)
    pdf.set_font('Arial', '', 9)
    pdf.multi_cell(0, 5, "Quick baseline, dimensionality reduction, perfect generalization.\n")
    
    # Conclusion
    pdf.ln(5)
    pdf.chapter_title('8. Conclusion')
    pdf.set_font('Arial', '', 10)
    pdf.multi_cell(0, 5,
        "All four models successfully learned to predict MBTI personality types:\n\n"
        "- XGBoost leads with 98.22% (best for production)\n"
        "- Random Forest at 97.57% (best balance)\n"
        "- Logistic Regression at 91.90% (best interpretability)\n"
        "- LDA at 90.56% (best baseline)\n\n"
        "Key Findings:\n"
        "1. Tree-based ensemble methods significantly outperform linear models\n"
        "2. All models achieve >98% top-3 accuracy\n"
        "3. Linear models show better generalization but lower raw accuracy\n"
        "4. ESFJ and ISFJ types are most challenging\n"
        "5. Feature importance is consistent, validating survey design\n\n"
        "All models trained with random_state=42 for reproducibility."
    )
    
    # Save
    output_path = 'Model_Comparison_Report.pdf'
    pdf.output(output_path, 'F')
    print(f"PDF generated: {os.path.abspath(output_path)}")

if __name__ == "__main__":
    try:
        create_pdf()
    except Exception as e:
        print(f"Error generating PDF: {e}")
        import traceback
        traceback.print_exc()
