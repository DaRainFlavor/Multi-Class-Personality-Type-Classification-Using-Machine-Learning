from fpdf import FPDF
import os

print("Starting PDF generation script...")

class PDF(FPDF):
    def header(self):
        # Arial bold 15
        self.set_font('Arial', 'B', 15)
        # Title
        self.cell(0, 10, 'Linear Discriminant Analysis (LDA) Classifier Report', 0, 1, 'C')
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
        # Background color (purple theme for LDA)
        self.set_fill_color(220, 200, 240)
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
    # Read metrics from evaluation report if available
    test_acc = "TBD"
    val_acc = "TBD"
    train_acc = "TBD"
    top3_acc = "TBD"
    macro_f1 = "TBD"
    
    # Try to read actual values from the evaluation report
    if os.path.exists('lda_evaluation_report.txt'):
        with open('lda_evaluation_report.txt', 'r') as f:
            content = f.read()
            # Parse actual values from the report
            for line in content.split('\n'):
                if 'Test Accuracy:' in line and 'Top' not in line:
                    test_acc = line.split('(')[1].split(')')[0] if '(' in line else "TBD"
                if 'Validation Accuracy:' in line:
                    val_acc = line.split('(')[1].split(')')[0] if '(' in line else "TBD"
                if 'Training Accuracy:' in line:
                    train_acc = line.split('(')[1].split(')')[0] if '(' in line else "TBD"
                if 'Top-3 Accuracy:' in line:
                    top3_acc = line.split('(')[1].split(')')[0] if '(' in line else "TBD"
    
    pdf = PDF()
    pdf.alias_nb_pages()
    pdf.add_page()
    
    # 1. Executive Summary
    pdf.chapter_title('1. Executive Summary')
    pdf.set_font('Arial', '', 11)
    pdf.multi_cell(0, 6, 
        "The Linear Discriminant Analysis (LDA) Classifier was trained to predict 16 MBTI personality types based on 60 survey questions. LDA finds linear combinations of features that best separate classes."
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
    pdf.cell(0, 6, "- Algorithm: Linear Discriminant Analysis", 0, 1)
    pdf.cell(10)
    pdf.cell(0, 6, "- Solver: SVD (Singular Value Decomposition)", 0, 1)
    pdf.cell(10)
    pdf.cell(0, 6, "- Feature Scaling: StandardScaler", 0, 1)
    pdf.cell(10)
    pdf.cell(0, 6, "- Discriminant Components: 15 (min of n_classes-1, n_features)", 0, 1)
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
        ('Test Accuracy', test_acc, 'Overall classification accuracy'),
        ('Validation Accuracy', val_acc, 'Validation set performance'),
        ('Train Accuracy', train_acc, 'Training set performance'),
        ('Top-3 Accuracy', top3_acc, 'Correct type in top 3 predictions'),
        ('Macro F1-Score', macro_f1, 'Balanced across all 16 classes'),
    ]
    
    for measure, score, notes in data:
        pdf.cell(50, 8, measure, 1)
        pdf.cell(40, 8, score, 1)
        pdf.cell(90, 8, notes, 1)
        pdf.ln()
    pdf.ln(5)

    # 4. Visualizations
    pdf.chapter_title('4. Visualizations')
    
    pdf.add_image_section('4.1 Confusion Matrix', 'figures/lda_confusion_matrix.png', width=140)
    pdf.add_page()  # New page for next large image
    pdf.add_image_section('4.2 Feature Importance', 'figures/lda_feature_importance.png', width=170)
    pdf.ln(5)
    pdf.add_image_section('4.3 Per-Class Accuracy', 'figures/lda_per_class_accuracy.png', width=170)

    # 5. Conclusion
    pdf.add_page()
    pdf.chapter_title('5. Conclusion')
    pdf.set_font('Arial', '', 11)
    pdf.multi_cell(0, 6, 
        f"LDA achieves {test_acc} test accuracy on the 16-class MBTI personality prediction task. "
        "As a classical linear classifier, LDA provides interpretable linear decision boundaries "
        "while also performing dimensionality reduction.\n\n"
        "Key Takeaways:\n"
        "- Uses same data splits as other models for fair comparison\n"
        "- Provides interpretable linear decision boundaries\n"
        "- Efficient computation via SVD solver\n"
        "- No hyperparameter tuning required"
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
