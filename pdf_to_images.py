import os
from pathlib import Path
from pdf2image import convert_from_path
from PIL import Image

# Define input and output directories
PDF_DIR = "pdfs/"  # Directory containing PDF files
OUTPUT_DIR_GS = "processed_imgs_gs/"  # Directory for grayscale images
OUTPUT_DIR_COLOR = "processed_imgs/"  # Directory for color images

# Ensure output directories exist
os.makedirs(OUTPUT_DIR_GS, exist_ok=True)
os.makedirs(OUTPUT_DIR_COLOR, exist_ok=True)

def convert_pdf_to_images(pdf_path, output_dir, grayscale=False, dpi=300):
    """
    Extracts pages from a PDF file and saves them as images with high quality.
    
    Args:
        pdf_path (str): Path to the PDF file.
        output_dir (str): Directory to save the images.
        grayscale (bool): Whether to convert the images to grayscale.
        dpi (int): Dots per inch for image quality (higher = better quality).
    """
    try:
        # Extract pages from the PDF file with the specified DPI (quality)
        print(f"Processing PDF: {pdf_path} with DPI={dpi}")
        pages = convert_from_path(pdf_path, dpi=dpi)
        
        # Iterate through each page and save as an image
        for page_num, page in enumerate(pages, start=1):
            # Convert to grayscale if specified
            if grayscale:
                page = page.convert("L")  # "L" mode is grayscale
            
            # Generate a proper naming convention for the image
            pdf_name = Path(pdf_path).stem  # Get the PDF file name without extension
            image_name = f"{pdf_name}_page_{page_num:03d}.png"  # Example: file_page_001.png
            
            # Save the image to the specified output directory
            output_path = os.path.join(output_dir, image_name)
            page.save(output_path, "PNG")
            print(f"Saved: {output_path}")
    except Exception as e:
        print(f"❌ Error processing {pdf_path}: {e}")

def process_pdfs(grayscale=False, dpi=300):
    """
    Processes all PDF files in the input directory and saves images to the appropriate output directory.
    
    Args:
        grayscale (bool): Whether to convert the images to grayscale.
        dpi (int): Dots per inch for image quality (higher = better quality).
    """
    # Determine the output directory based on grayscale flag
    output_dir = OUTPUT_DIR_GS if grayscale else OUTPUT_DIR_COLOR
    
    # Get a list of all PDF files in the input directory
    pdf_files = [f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")]
    
    if not pdf_files:
        print(f"No PDF files found in {PDF_DIR}")
        return
    
    # Process each PDF file
    for pdf_file in pdf_files:
        pdf_path = os.path.join(PDF_DIR, pdf_file)
        convert_pdf_to_images(pdf_path, output_dir, grayscale=grayscale, dpi=dpi)

if __name__ == "__main__":
    # Part 1: Convert PDFs to grayscale images with high quality
    print("=== Converting PDFs to Grayscale Images (High Quality) ===")
    process_pdfs(grayscale=True, dpi=300)
    
    # Part 2: Convert PDFs to color images with high quality
    print("\n=== Converting PDFs to Color Images (High Quality) ===")
    process_pdfs(grayscale=False, dpi=300)