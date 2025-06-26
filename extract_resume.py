import fitz  # PyMuPDF
from pathlib import Path

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

if __name__ == "__main__":
    pdf_path = Path("resume.pdf")  # Make sure your resume is named this
    if not pdf_path.exists():
        print("❌ File not found!")
    else:
        text = extract_text_from_pdf(pdf_path)
        print("\n--- ✅ Extracted Text Start ---\n")
        print(text[:1500])  # Show first 1500 characters
        print("\n--- ✅ Text End ---")
