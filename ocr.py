import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import os
import io

# Set Tesseract path (update if installed elsewhere)
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\acer\Tesseract-OCR\tesseract.exe'

def pdf_to_text(input_pdf, output_folder):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Open PDF
    print(f"Converting PDF: {input_pdf}")
    pdf_document = fitz.open(input_pdf)
    
    all_text = []
    
    # Process each page
    for page_num in range(len(pdf_document)):
        print(f"Processing page {page_num + 1}/{len(pdf_document)}")
        
        page = pdf_document[page_num]
        
        # Convert page to image
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better quality
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        
        # Extract text from image
        text = pytesseract.image_to_string(img)
        all_text.append(f"\n--- Page {page_num + 1} ---\n\n{text}")
    
    pdf_document.close()
    
    # Write all text to a single file
    output_file = os.path.join(output_folder, f"{os.path.basename(input_pdf)}.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(all_text))
    
    print(f"Text extracted and saved to: {output_file}")

if __name__ == "__main__":
    input_pdf = r"data\bhagavad-gita-as-it-is.pdf"
    output_folder = "data"
    
    pdf_to_text(input_pdf, output_folder)
