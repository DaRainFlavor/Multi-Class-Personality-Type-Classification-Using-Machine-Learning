from fpdf import FPDF
import os

print("Starting PDF generation script...")

class PDF(FPDF):
    def header(self):
        # Arial bold 15
        self.set_font('Arial', 'B', 15)
        # Title
        self.cell(0, 10, 'Logistic Regression Classifier Report', 0, 1, 'C')
        self.set_font('Arial', '', 10)
        self.cell(0, 10, 'Project: Multi-Class Classification of Personality Types (MBTI)', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        # Position at 1.5 cm from bottom
        self.set_y(-15)
        # Arial italic 8
        self.set_font('Arial', 'I', 8)
        # Page number
        self.cell(0, 10, 'Page ' + str(self.page_no()) + '/{nb}', 0, 0, 'C')

    def chapter_title(self, label):
        # Arial 12
        self.set_font('Arial', 'B', 12)
        # Background color (green theme for logistic regression)
        self.set_fill_color(200, 235, 200)
        # Title
        self.cell(0, 6, label, 0, 1, 'L', 1)
        # Line break
        self.ln(4)

    def chapter_body(self, txt):
        # Times 12
        self.set_font('Times', '', 12)
        # Output justified text
        self.multi_cell(0, 10, txt)
        # Line break
        self.ln()

    def add_image_section(self, title, image_path, width=150):
        self.set_font('Arial', 'B', 11)
        self.cell(0, 10, title, 0, 1)
        if os.path.exists(image_path):
            self.image(image_path, w=width)
            self.ln(5)
        else:
            self.set_font('Arial', 'I', 10)
            self.cell(0, 10, f"[Image not found: {image_path}]", 0, 1)

def create_pdf():
    pdf = PDF()
    pdf.alias_nb_pages()
    pdf.add_page()
    
    # 1. Executive Summary
    pdf.chapter_title('1. Executive Summary')
    pdf.set_font('Arial', '', 11)
    pdf.multi_cell(0, 6, 
        "The Logistic Regression Classifier (Multinomial/Softmax) was trained to predict 16 MBTI personality types based on 60 survey questions."
    )
    pdf.ln(5)

    # 2. Model Configuration
    pdf.chapter_title('2. Model Configuration')
    pdf.set_font('Arial', 'B', 10)
    pdf.cell(0, 6, "Data Split:", 0, 1)
    pdf.set_font('Arial', '', 10)
    pdf.cell(10)
    pdf.cell(0, 6, "- Training: 70% (42,023 samples)", 0, 1)
    pdf.cell(10)
    pdf.cell(0, 6, "- Validation: 15% (8,976 samples)", 0, 1)
    pdf.cell(10)
    pdf.cell(0, 6, "- Test: 15% (9,000 samples)", 0, 1)
    pdf.ln(2)
    
    pdf.set_font('Arial', 'B', 10)
    pdf.cell(0, 6, "Hyperparameters:", 0, 1)
    pdf.set_font('Arial', '', 10)
    pdf.cell(10)
    pdf.cell(0, 6, "- Algorithm: Logistic Regression (Multinomial)", 0, 1)
    pdf.cell(10)
    pdf.cell(0, 6, "- Solver: L-BFGS", 0, 1)
    pdf.cell(10)
    pdf.cell(0, 6, "- Regularization (C): 1.0", 0, 1)
    pdf.cell(10)
    pdf.cell(0, 6, "- Max Iterations: 1000 (converged in 24)", 0, 1)
    pdf.cell(10)
    pdf.cell(0, 6, "- Feature Scaling: StandardScaler", 0, 1)
    pdf.ln(5)

    # 3. Performance Metrics
    pdf.chapter_title('3. Performance Metrics')
    
    # Table Header
    pdf.set_font('Arial', 'B', 10)
    pdf.cell(50, 8, 'Measure', 1)
    pdf.cell(40, 8, 'Score', 1)
    pdf.cell(90, 8, 'Notes', 1)
    pdf.ln()
    
    # Table Rows with actual values
    pdf.set_font('Arial', '', 10)
    data = [
        ('Test Accuracy', '91.90%', 'Overall classification accuracy'),
        ('Validation Accuracy', '92.12%', 'Validation set performance'),
        ('Train Accuracy', '92.35%', 'Training set performance'),
        ('Top-3 Accuracy', '98.62%', 'Correct type in top 3 predictions'),
        ('Macro F1-Score', '0.9189', 'Balanced across all 16 classes'),
    ]
    
    for measure, score, notes in data:
        pdf.cell(50, 8, measure, 1)
        pdf.cell(40, 8, score, 1)
        pdf.cell(90, 8, notes, 1)
        pdf.ln()
    pdf.ln(5)

    # 4. Visualizations
    pdf.chapter_title('4. Visualizations')
    
    pdf.add_image_section('4.1 Confusion Matrix', 'figures/lr_confusion_matrix.png', width=140)
    pdf.add_page()  # New page for next large image
    pdf.add_image_section('4.2 Feature Importance', 'figures/lr_feature_importance.png', width=170)
    pdf.ln(5)
    pdf.add_image_section('4.3 Per-Class Accuracy', 'figures/lr_per_class_accuracy.png', width=170)

    # 5. Conclusion
    pdf.add_page()
    pdf.chapter_title('5. Conclusion')
    pdf.set_font('Arial', '', 11)
    pdf.multi_cell(0, 6, 
        "Logistic Regression achieves 91.90% test accuracy on the 16-class MBTI personality prediction task. "
        "Its coefficient-based feature importance provides clear, interpretable insights into which survey questions "
        "are most predictive of personality types.\n\n"
        "Key Takeaways:\n"
        "- Strong generalization (train-test gap only 0.45%)\n"
        "- High top-k accuracy (98.62% for top-3)\n"
        "- Interpretable coefficients for feature analysis\n"
        "- Model simplicity and faster training time"
    )
    
    # Save
    output_path = 'Model_Report.pdf'
    pdf.output(output_path, 'F')
    print(f"PDF generated: {os.path.abspath(output_path)}")

if __name__ == "__main__":
    try:
        create_pdf()
    except Exception as e:
        print(f"Error generating PDF: {e}")
