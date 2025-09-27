import os
import pytesseract
import openai
from PIL import Image

# Define directories
PROCESSED_IMGS_GS_DIR = "processed_imgs_gs/"  # Directory for grayscale images
PROCESSED_IMGS_DIR = "processed_imgs/"  # Directory for color images
RESULTS_TESS_CORRECTION_DIR = "results_nonexplicit/tess_correction/"  # Directory for Tesseract + OpenAI correction results
RESULTS_VISION_DIR = "results_nonexplicit/vision/"  # Directory for OpenAI Vision API results

# Ensure output directories exist
os.makedirs(RESULTS_TESS_CORRECTION_DIR, exist_ok=True)
os.makedirs(RESULTS_VISION_DIR, exist_ok=True)

# OpenAI API key (replace with your actual key)
openai.api_key = "your_openai_api_key"

def correct_text_with_openai(text):
    """
    Use OpenAI's API to correct the text.
    """
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=f"Correct the following OCR text:\n\n{text}",
            max_tokens=1000,
            temperature=0.5
        )
        return response.choices[0].text.strip()
    except Exception as e:
        print(f"❌ Error using OpenAI API for text correction: {e}")
        return text

def transcribe_with_openai_vision(image_path):
    """
    Use OpenAI's Vision API to transcribe text from an image.
    """
    try:
        with open(image_path, "rb") as image_file:
            response = openai.Image.create(
                file=image_file,
                purpose="transcription"
            )
        return response["data"]["text"]
    except Exception as e:
        print(f"❌ Error using OpenAI Vision API: {e}")
        return ""

def process_images_with_tesseract(image_files, input_dir, output_dir):
    """
    Process images with Tesseract OCR and correct the text using OpenAI.
    """
    for image_file in image_files:
        image_path = os.path.join(input_dir, image_file)
        try:
            print(f"Processing with Tesseract: {image_file}")
            # Perform OCR using Tesseract with PSM 3
            ocr_text = pytesseract.image_to_string(Image.open(image_path), config="--psm 3")
            print(f"OCR Text: {ocr_text[:100]}...")  # Show a snippet of the OCR text
            
            # Correct the text using OpenAI
            corrected_text = correct_text_with_openai(ocr_text)
            
            # Save the corrected text to a .txt file
            output_file = os.path.join(output_dir, f"{Path(image_file).stem}_corrected.txt")
            with open(output_file, "w") as f:
                f.write(corrected_text)
            print(f"Saved corrected text to: {output_file}")
        except Exception as e:
            print(f"❌ Error processing {image_file} with Tesseract: {e}")

def process_images_with_openai_vision(image_files, input_dir, output_dir):
    """
    Process images with OpenAI Vision API to transcribe text.
    """
    for image_file in image_files:
        image_path = os.path.join(input_dir, image_file)
        try:
            print(f"Processing with OpenAI Vision: {image_file}")
            # Transcribe text using OpenAI Vision API
            transcribed_text = transcribe_with_openai_vision(image_path)
            
            # Save the transcribed text to a .txt file
            output_file = os.path.join(output_dir, f"{Path(image_file).stem}_vision.txt")
            with open(output_file, "w") as f:
                f.write(transcribed_text)
            print(f"Saved transcribed text to: {output_file}")
        except Exception as e:
            print(f"❌ Error processing {image_file} with OpenAI Vision: {e}")

if __name__ == "__main__":
    while True:
        print("\n=== OCR Processing Loop ===")
        
        # Step 1: Get grayscale images to process
        print("\nEnter the names of the grayscale images to process (comma-separated):")
        grayscale_images = input("Grayscale Images: ").split(",")
        grayscale_images = [img.strip() for img in grayscale_images if img.strip()]
        
        # Step 2: Process grayscale images with Tesseract and OpenAI correction
        process_images_with_tesseract(grayscale_images, PROCESSED_IMGS_GS_DIR, RESULTS_TESS_CORRECTION_DIR)
        
        # Step 3: Get color images to process
        print("\nEnter the names of the color images to process (comma-separated):")
        color_images = input("Color Images: ").split(",")
        color_images = [img.strip() for img in color_images if img.strip()]
        
        # Step 4: Process color images with OpenAI Vision API
        process_images_with_openai_vision(color_images, PROCESSED_IMGS_DIR, RESULTS_VISION_DIR)
        
        # Ask if the user wants to process more images
        continue_choice = input("\nDo you want to process more images? (yes/no): ").strip().lower()
        if continue_choice != "yes":
            print("Exiting the OCR processing loop.")
            break