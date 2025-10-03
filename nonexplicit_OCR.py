import os
import pytesseract
import openai
from PIL import Image
from pathlib import Path
import base64


# Define directories
PROCESSED_IMGS_GS_DIR = "processed_imgs_gs/"  # Directory for grayscale images
PROCESSED_IMGS_DIR = "processed_imgs/"  # Directory for color images
RESULTS_TESS_CORRECTION_DIR = "results_nonexplicit/tess_correction/"  # Directory for Tesseract + OpenAI correction results
RESULTS_VISION_DIR = "results_nonexplicit/vision/"  # Directory for Tesseract OCR results (color images)

# Define the images
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
            max_tokens=3000,
            temperature=0.1  # Low temperature for more consistent corrections
        )
        
        # Extract the corrected text from the response
        corrected_text = response.choices[0].message.content.strip()

        return corrected_text

    except Exception as e:
        print(f"❌ Error calling OpenAI API: {e}")
        return text  # Return the original text if correction fails

def process_images_with_tesseract_and_ai(image_files, input_dir="processed_imgs_gs/", output_dir="results_nonexplicit/tess_correction/"):
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

def encode_image(image_path):
    """
    Encode an image file to a base64 string.
           
    Args:
        image_path (str): Path to the image file.
                
    Returns:
        str: Base64 encoded string of the image.
    """
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return None

def transcribe_with_vision_api(image_path, api_key):
    """
    Transcribe text from an image using the OpenAI Vision API.

    Args:
        image_path (str): Path to the image file.
        api_key (str): OpenAI API key.

    Returns:
        tuple: (transcribed_text, usage_info)
    """
    try:
        # Initialize OpenAI client
        openai.api_key = api_key

        # Encode the image
        base64_image = encode_image(image_path)
        if not base64_image:
            return None, None

        # Determine image format
        image_format = Path(image_path).suffix.lower()
        if image_format not in ['.jpg', '.jpeg', '.png', '.gif', '.webp']:
            print(f"Warning: {image_format} may not be supported by Vision API")

        # Send request to OpenAI Vision API
        response = openai.chat.completions.create(
            model="gpt-4o",  # Using GPT-4o for vision capabilities
            messages=[
                {
                    "role": "user",
                    "content": f"""Please transcribe all the text visible in this image. The text may be in any language including English, Spanish, French, German, Chinese, Japanese, Korean, Arabic, Hindi, Tamil, or other languages.

Instructions:
1. Extract ALL text from the image, preserving the original layout, structure, and content.
2. Maintain line breaks and paragraph structure.
3. Do not add any commentary or interpretation.
4. If text is unclear or partially obscured, transcribe what you can see.
5. Preserve original spelling and formatting in the original language.
6. Include headers, titles, dates, and all visible text elements.
7. If the text is in a non-Latin script (like Arabic, Chinese, Tamil, etc.), transcribe it exactly as written.
8. Do not translate the text - only transcribe it.

Image data: data:image/{image_format[1:]};base64,{base64_image}"""
                }
            ],
            max_tokens=4000,
            temperature=0.1  # Low temperature for consistent transcription
        )

        # Extract usage information
        usage = response.usage
        usage_info = {
            'prompt_tokens': usage.prompt_tokens,
            'completion_tokens': usage.completion_tokens,
            'total_tokens': usage.total_tokens
        }

        # Extract the transcribed text
        transcribed_text = response.choices[0].message.content.strip()
        return transcribed_text, usage_info

    except Exception as e:
        print(f"❌ Error calling OpenAI Vision API: {e}")
        return None, None

def main():
    """
    Main function to process images with both Tesseract + correction and OpenAI Vision.
    """
    # Define directories
    PROCESSED_IMGS_GS_DIR = "processed_imgs_gs/"  # Directory for grayscale images
    PROCESSED_IMGS_DIR = "processed_imgs/"  # Directory for color images
    RESULTS_TESS_CORRECTION_DIR = "results_nonexplicit/tess_correction/"  # Directory for Tesseract + OpenAI correction results
    RESULTS_VISION_DIR = "results_nonexplicit/vision/"  # Directory for OpenAI Vision results

    # Ensure output directories exist
    os.makedirs(RESULTS_TESS_CORRECTION_DIR, exist_ok=True)
    os.makedirs(RESULTS_VISION_DIR, exist_ok=True)

    # Define the images to process directly in the script
    grayscale_images = ["AintVol1No7_page_003.png", "OOBVol1No1_page_006.png", "BabeVol1No2_page_012.png"]  # Replace with your grayscale image file names
    color_images = ["AintVol1No7_page_003.png", "OOBVol1No1_page_006.png", "BabeVol1No2_page_012.png"]  # Replace with your color image file names

    # Retrieve the API key from the environment variable
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("API key not found. Please set the OPENAI_API_KEY environment variable.")

    # Process grayscale images with Tesseract + AI correction
    print("Starting Tesseract + Correction processing for grayscale images...")
    process_images_with_tesseract_and_ai(grayscale_images, PROCESSED_IMGS_GS_DIR, RESULTS_TESS_CORRECTION_DIR)

    # Process color images with OpenAI Vision
    print("Starting OpenAI Vision processing for color images...")
    for image_file in color_images:
        image_path = os.path.join(PROCESSED_IMGS_DIR, image_file)
        transcribed_text, usage_info = transcribe_with_vision_api(image_path, api_key)
        if transcribed_text:
            # Save the transcribed text to a .txt file
            output_file = os.path.join(RESULTS_VISION_DIR, f"{Path(image_file).stem}_vision.txt")
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(transcribed_text)
            print(f"Saved Vision OCR text to: {output_file}")

if __name__ == "__main__":
    main()

def calculate_cost(usage_info, model="gpt-4o"):
    """
    Calculate estimated cost based on token usage.
    
    Args:
        usage_info (dict): Token usage information
        model (str): Model used for the API call
        
    Returns:
        dict: Cost breakdown
    """
    # Pricing per 1K tokens (as of 2024)
    pricing = {
        "gpt-3.5-turbo": {
            "input": 0.0005,   # $0.50 per 1M tokens
            "output": 0.0015   # $1.50 per 1M tokens
        },
        "gpt-4o": {
            "input": 0.005,    # $5.00 per 1M tokens
            "output": 0.015    # $15.00 per 1M tokens
        },
        "gpt-5-mini": {
            "input": 0.0025,     #$0.25 per 1M tokens
            "output": 0.002     # $2.00 per 1M tokens   
        }
    }
    
    if model not in pricing:
        model = "gpt-4o"  # Default fallback
    
    input_cost = (usage_info['prompt_tokens'] / 1000) * pricing[model]["input"]
    output_cost = (usage_info['completion_tokens'] / 1000) * pricing[model]["output"]
    total_cost = input_cost + output_cost
    
    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost,
        "model": model
    }

def print_usage_summary(usage_info, cost_info):
    """
    Print token usage and cost summary to terminal.
    
    Args:
        usage_info (dict): Token usage information
        cost_info (dict): Cost breakdown
    """
    print("\n" + "=" * 50)
    print("OPENAI VISION API USAGE SUMMARY")
    print("=" * 50)
    print(f"Model: {cost_info['model']}")
    print(f"Prompt Tokens:  {usage_info['prompt_tokens']:,}")
    print(f"Output Tokens:  {usage_info['completion_tokens']:,}")
    print(f"Total Tokens:   {usage_info['total_tokens']:,}")
    print("-" * 50)
    print(f"Input Cost:     ${cost_info['input_cost']:.4f}")
    print(f"Output Cost:    ${cost_info['output_cost']:.4f}")
    print(f"Total Cost:     ${cost_info['total_cost']:.4f}")
    print("=" * 50)

if __name__ == "__main__":
    print("\n=== OCR Processing ===")
    
    # Step 1: Process grayscale images with Tesseract and OpenAI correction
    print("\nProcessing grayscale images with Tesseract and OpenAI correction...")
    process_images_with_tesseract_and_ai(grayscale_images, PROCESSED_IMGS_GS_DIR, RESULTS_TESS_CORRECTION_DIR)
    
    # Step 2: Process color images with Tesseract OCR (no AI correction)
    print("\nProcessing color images with Tesseract OCR...")
    process_images_with_tesseract_and_ai(color_images, PROCESSED_IMGS_DIR, RESULTS_VISION_DIR)
    
    print("\nProcessing complete.")