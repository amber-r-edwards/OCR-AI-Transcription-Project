import os
import shutil
import tempfile
from pathlib import Path
from pdf2image import convert_from_path
from PIL import Image

# Define input and output directories
PDF_DIR = "pdfs/"  # Directory containing PDF files
OUTPUT_DIR_COLOR = "processed_imgs/"  # Directory for selected color images
OUTPUT_DIR_GS = "processed_imgs_gs/"  # Directory for selected grayscale images

# Ensure output directories exist
os.makedirs(OUTPUT_DIR_COLOR, exist_ok=True)
os.makedirs(OUTPUT_DIR_GS, exist_ok=True)

def convert_pdf_to_temp_images(pdf_path, temp_dir, dpi=150):
    """
    Extracts pages from a PDF file and saves them as images in a temporary directory.
    
    Args:
        pdf_path (str): Path to the PDF file.
        temp_dir (str): Temporary directory to save the images.
        dpi (int): Dots per inch for image quality (higher = better quality).
    """
    try:
        # Extract pages from the PDF file with the specified DPI (quality)
        print(f"Processing PDF: {pdf_path} with DPI={dpi}")
        pages = convert_from_path(pdf_path, dpi=dpi)
        
        # Iterate through each page and save as an image
        for page_num, page in enumerate(pages, start=1):
            # Generate a proper naming convention for the image
            pdf_name = Path(pdf_path).stem  # Get the PDF file name without extension
            image_name = f"{pdf_name}_page_{page_num:03d}.png"  # Example: file_page_001.png
            
            # Save the image to the temporary directory
            output_path = os.path.join(temp_dir, image_name)
            page.save(output_path, "PNG")
            print(f"Saved: {output_path}")
    except Exception as e:
        print(f"❌ Error processing {pdf_path}: {e}")

def process_single_pdf():
    """
    Processes one PDF at a time, saves images to a temporary directory, and allows the user
    to select whether to process the images in grayscale or color.
    """
    # Get a list of all PDF files in the input directory
    pdf_files = [f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")]
    
    if not pdf_files:
        print(f"No PDF files found in {PDF_DIR}")
        return
    
    # Display the list of PDFs and let the user select one
    print("Available PDFs:")
    for idx, pdf_file in enumerate(pdf_files, start=1):
        print(f"{idx}. {pdf_file}")
    
    try:
        choice = int(input("Enter the number of the PDF to process: "))
        if choice < 1 or choice > len(pdf_files):
            print("Invalid choice. Exiting.")
            return
    except ValueError:
        print("Invalid input. Exiting.")
        return
    
    # Get the selected PDF file
    selected_pdf = pdf_files[choice - 1]
    pdf_path = os.path.join(PDF_DIR, selected_pdf)
    print(f"Selected PDF: {selected_pdf}")
    
    # Ask the user whether to process in grayscale or color
    while True:
        grayscale_choice = input("Do you want to process the images in grayscale? (yes/no): ").strip().lower()
        if grayscale_choice in ["yes", "no"]:
            grayscale = grayscale_choice == "yes"
            break
        else:
            print("Invalid input. Please enter 'yes' or 'no'.")
    
    # Create a temporary directory for processing
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Temporary directory created: {temp_dir}")
        
        # Convert the PDF to images in the temporary directory
        convert_pdf_to_temp_images(pdf_path, temp_dir, dpi=150)
        
        # Display the images in the temporary directory
        temp_images = [f for f in os.listdir(temp_dir) if f.endswith(".png")]
        print("\nExtracted Pages:")
        for idx, image_file in enumerate(temp_images, start=1):
            print(f"{idx}. {image_file}")
        
        # Ask the user which pages to move based on grayscale choice
        if grayscale:
            print("\nEnter the numbers of the pages to move to the processed_imgs_gs/ folder (comma-separated):")
            output_dir = OUTPUT_DIR_GS
        else:
            print("\nEnter the numbers of the pages to move to the processed_imgs/ folder (comma-separated):")
            output_dir = OUTPUT_DIR_COLOR
        
        try:
            selected_pages = input("Pages: ").split(",")
            selected_pages = [int(page.strip()) for page in selected_pages if page.strip()]
        except ValueError:
            print("Invalid input. Exiting.")
            return
        
        # Move the selected pages to the appropriate folder
        for page_num in selected_pages:
            if page_num < 1 or page_num > len(temp_images):
                print(f"Invalid page number: {page_num}. Skipping.")
                continue
            
            image_file = temp_images[page_num - 1]
            src_path = os.path.join(temp_dir, image_file)
            dest_path = os.path.join(output_dir, image_file)
            
            if grayscale:
                # Convert to grayscale before moving
                with Image.open(src_path) as img:
                    grayscale_img = img.convert("L")
                    grayscale_img.save(dest_path)
                print(f"Moved to Grayscale Folder: {image_file}")
            else:
                shutil.move(src_path, dest_path)
                print(f"Moved to Color Folder: {image_file}")
        
        print("\nProcessing complete. Selected pages have been moved.")

if __name__ == "__main__":
    process_single_pdf()