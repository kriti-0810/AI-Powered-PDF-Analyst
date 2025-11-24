import os
from PyPDF2 import PdfReader

def load_pdf_files(pdf_paths):
    """
    Takes a list of PDF file paths.
    Returns extracted text from all PDFs + metadata.
    
    Output format:
    [
        {
            "filename": "...",
            "page_number": 1,
            "text": "extracted text..."
        },
        ...
    ]
    """
    
    extracted_data = []

    for pdf_path in pdf_paths:
        if not os.path.exists(pdf_path):
            print(f"❌ File not found: {pdf_path}")
            continue

        try:
            reader = PdfReader(pdf_path)
            filename = os.path.basename(pdf_path)

            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                cleaned_text = text.replace("\n", " ").strip()

                extracted_data.append({
                    "filename": filename,
                    "page_number": i + 1,
                    "text": cleaned_text
                })

        except Exception as e:
            print(f"Error reading {pdf_path}: {e}")

    return extracted_data
