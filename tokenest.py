import os
import base64
from pathlib import Path
import tiktoken
from PIL import Image

def resize_image(image_path, max_width=1024, max_height=1024):
    """
    Resize the image to reduce its size while maintaining aspect ratio.
    """
    with Image.open(image_path) as img:
        img.thumbnail((max_width, max_height))
        img.save(image_path)  # Overwrite the original image with the resized version

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

def estimate_tokens(model, prompt, base64_image):
    """
    Estimate the number of tokens used in the request.

    Args:
        model (str): The OpenAI model (e.g., "gpt-3.5-turbo").
        prompt (str): The prompt content.
        base64_image (str): The base64-encoded image data.

    Returns:
        dict: A breakdown of token usage.
    """
    # Initialize the tokenizer for the specified model
    encoding = tiktoken.encoding_for_model(model)

    # Tokenize the prompt and base64 image data
    prompt_tokens = len(encoding.encode(prompt))
    image_tokens = len(encoding.encode(base64_image))

    # Calculate total tokens
    total_tokens = prompt_tokens + image_tokens

    return {
        "prompt_tokens": prompt_tokens,
        "image_tokens": image_tokens,
        "total_tokens": total_tokens
    }

def main():
    """
    Main function to estimate token usage for OpenAI Vision API requests.
    """
    # Define the model
    model = "gpt-3.5-turbo"

    # Define the image path
    image_path = "processed_imgs/AintVol1No7_page_003.png"  # Replace with your image path

    # Resize the image to reduce its size
    resize_image(image_path)

    # Encode the image to base64
    base64_image = encode_image(image_path)
    if not base64_image:
        print("Failed to encode the image.")
        return

    # Define the prompt
    prompt = """Please transcribe all visible text from this image. Preserve layout, line breaks, and original spelling. Do not translate or interpret."""

    # Estimate token usage
    token_usage = estimate_tokens(model, prompt, f"data:image/png;base64,{base64_image}")

    # Print the token breakdown
    print("Token Usage Breakdown:")
    print(f"Prompt Tokens: {token_usage['prompt_tokens']}")
    print(f"Image Tokens: {token_usage['image_tokens']}")
    print(f"Total Tokens: {token_usage['total_tokens']}")

if __name__ == "__main__":
    main()