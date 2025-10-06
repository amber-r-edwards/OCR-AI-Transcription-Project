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
    return f"Text to correct:\n{text}\n\nCorrected text:"


def resize_and_convert_to_jpeg(image_path, max_width=1024, max_height=1024):
    """
    Resize the image to reduce its size while maintaining aspect ratio.
    Convert the image to JPEG format to further reduce file size.
    """
    with Image.open(image_path) as img:
        img.thumbnail((max_width, max_height))  # Resize while maintaining aspect ratio
        jpeg_path = image_path.replace(".png", ".jpg")  # Save as JPEG
        img = img.convert("RGB")  # Ensure compatibility with JPEG
        img.save(jpeg_path, "JPEG", quality=85)  # Adjust quality as needed
        return jpeg_path


def encode_image(image_path):
    """
    Encode an image file to a base64 string.
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
        client = OpenAI(api_key=api_key)
        
        # Resize and convert the image to JPEG
        resized_image_path = resize_and_convert_to_jpeg(image_path)

        # Encode the resized image to base64
        base64_image = encode_image(resized_image_path)
        if not base64_image:
            return None, None

        # Determine image format
        image_format = Path(resized_image_path).suffix.lower()
        if image_format not in ['.jpg', '.jpeg', '.png', '.gif', '.webp']:
            print(f"Warning: {image_format} may not be supported by Vision API")

        # Send request to OpenAI Vision API using the correct format
        response = client.chat.completions.create(
            model="gpt-4o",  # Use gpt-4o or gpt-4-vision-preview for vision capabilities
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Please transcribe all visible text from this image. Preserve layout, line breaks, and original spelling. Do not translate or interpret."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/{image_format[1:]};base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=4000,
            temperature=0.1
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
        
        # Print token usage and estimated cost
        print(f"🔹 Token Usage: Prompt = {usage_info['prompt_tokens']}, Completion = {usage_info['completion_tokens']}, Total = {usage_info['total_tokens']}")
        
        # Calculate cost (adjust rates based on current OpenAI pricing)
        # For gpt-4o: $5 per 1M input tokens, $15 per 1M output tokens (example rates)
        cost_per_input_token = 5.00 / 1_000_000
        cost_per_output_token = 15.00 / 1_000_000
        estimated_cost = (usage_info['prompt_tokens'] * cost_per_input_token) + (usage_info['completion_tokens'] * cost_per_output_token)
        print(f"💰 Estimated Cost: ${estimated_cost:.6f}")
        
        return transcribed_text, usage_info
    
    except Exception as e:
        print(f"❌ Error calling OpenAI Vision API: {e}")
        return None, None
    
def main():
    """
    Main function to process images with both Tesseract + correction and OpenAI Vision.
    """
    # Ensure output directories exist
    os.makedirs(RESULTS_TESS_CORRECTION_DIR, exist_ok=True)
    os.makedirs(RESULTS_VISION_DIR, exist_ok=True)

    # Retrieve the API key from the environment variable
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("API key not found. Please set the OPENAI_API_KEY environment variable.")

    # Process grayscale images with Tesseract + AI correction
    print("Starting Tesseract + Correction processing for grayscale images...")
    for image_file in grayscale_images:
        image_path = os.path.join(PROCESSED_IMGS_GS_DIR, image_file)
        try:
            # Perform OCR using Tesseract
            ocr_text = pytesseract.image_to_string(Image.open(image_path), config="--psm 3")
            print(f"OCR Text: {ocr_text[:100]}...")  # Show a snippet of the OCR text

            # Correct the text using OpenAI
            corrected_text = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert at correcting OCR text from historical documents."},
                    {"role": "user", "content": create_correction_prompt(ocr_text)}
                ],
                max_tokens=3000,
                temperature=0.1
            ).choices[0].message.content.strip()

            # Save the corrected text to a .txt file
            output_file = os.path.join(RESULTS_TESS_CORRECTION_DIR, f"{Path(image_file).stem}_corrected.txt")
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(corrected_text)
            print(f"Saved corrected text to: {output_file}")
        except Exception as e:
            print(f"❌ Error processing {image_file} with Tesseract and AI: {e}")

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