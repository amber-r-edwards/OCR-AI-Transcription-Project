import os
import pytesseract
import openai
from PIL import Image
from pathlib import Path

# Define directories
PROCESSED_IMGS_GS_DIR = "processed_imgs_gs/"  # Directory for grayscale images
PROCESSED_IMGS_DIR = "processed_imgs/"  # Directory for color images
RESULTS_TESS_CORRECTION_DIR = "results_nonexplicit/tess_correction/"  # Directory for Tesseract + OpenAI correction results
RESULTS_VISION_DIR = "results_nonexplicit/vision/"  # Directory for Tesseract OCR results (color images)

# Ensure output directories exist
os.makedirs(RESULTS_TESS_CORRECTION_DIR, exist_ok=True)
os.makedirs(RESULTS_VISION_DIR, exist_ok=True)


# Define the images to process directly in the script
grayscale_images = ["AintVol1No7_page_003.png", "OOBVol1No1_page_006.png", "BabeVol1No2_page_012.png"]  # Replace with your grayscale image file names
color_images = ["AintVol1No7_page_003.png", "OOBVol1No1_page_006.png", "BabeVol1No2_page_012.png"]  # Replace with your color image file names

def create_correction_prompt(text):
    """
    Create a prompt for OpenAI to correct the text.
    """
    prompt = f"""Please correct the following text that was extracted from a historical document using OCR. 
The text may contain OCR errors, missing punctuation, or formatting issues.

Please:
1. Fix obvious OCR errors (like '0' instead of 'O', '1' instead of 'l', etc.)
2. Add appropriate punctuation and capitalization
3. Fix spacing and line breaks where needed
4. Preserve the words as they are on the page, do not add in any additional context.


Text to correct:
{text}

Corrected text:"""
    return prompt

def correct_text_with_ai(text):
    """
    Send text to OpenAI API for correction.
    """
    try:
         # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")  # Load API key from environment variable
        if not api_key:
            raise ValueError("API key not found. Please set the OPENAI_API_KEY environment variable.")
        client = openai  # Use the OpenAI library as the client
        openai.api_key = api_key  # Set the API key

        # Create the correction prompt
        prompt = create_correction_prompt(text)
        
        # Send request to OpenAI
        response = client.chat.completions.create(
            model="gpt-3.5-turbo", #model selected: GPT-3.5-turbo
            messages=[
                {"role": "system", "content": "You are an expert at correcting OCR text from historical documents. Focus on accuracy and preserving the exact words on the pages."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=2000,
            temperature=0.1  # Low temperature for more consistent corrections
        )
        
        # Extract the corrected text from the response
        corrected_text = response["choices"][0]["message"]["content"].strip()
        return corrected_text
    except Exception as e:
        print(f"❌ Error calling OpenAI API: {e}")
        return text  # Return the original text if correction fails

def process_images_with_tesseract_and_ai(image_files, input_dir, output_dir):
    """
    Process images with Tesseract OCR and correct the text using OpenAI.
    """
    for image_file in image_files:
        image_path = os.path.join(input_dir, image_file)
        try:
            print(f"Processing with Tesseract: {image_file}")
            # Perform OCR using Tesseract
            ocr_text = pytesseract.image_to_string(Image.open(image_path), config="--psm 3")
            print(f"OCR Text: {ocr_text[:100]}...")  # Show a snippet of the OCR text
            
            # Correct the text using OpenAI
            corrected_text = correct_text_with_ai(ocr_text)
            
            # Save the corrected text to a .txt file
            output_file = os.path.join(output_dir, f"{Path(image_file).stem}_corrected.txt")
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(corrected_text)
            print(f"Saved corrected text to: {output_file}")
        except Exception as e:
            print(f"❌ Error processing {image_file} with Tesseract and AI: {e}")

if __name__ == "__main__":
    print("\n=== OCR Processing ===")
    
    # Step 1: Process grayscale images with Tesseract and OpenAI correction
    print("\nProcessing grayscale images with Tesseract and OpenAI correction...")
    process_images_with_tesseract_and_ai(grayscale_images, PROCESSED_IMGS_GS_DIR, RESULTS_TESS_CORRECTION_DIR)
    
    # Step 2: Process color images with Tesseract OCR (no AI correction)
    print("\nProcessing color images with Tesseract OCR...")
    process_images_with_tesseract_and_ai(color_images, PROCESSED_IMGS_DIR, RESULTS_VISION_DIR)
    
    print("\nProcessing complete.")