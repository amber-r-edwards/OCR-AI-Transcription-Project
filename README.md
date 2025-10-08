# OCR AI Project: Historical Document Transcription

This project uses OCR (Optical Character Recognition) and AI to transcribe historical documents from the women's liberation movement. It provides two main processing methods: Tesseract OCR with AI correction for grayscale images, and OpenAI's Vision API for color images.

## 📋 Project Overview

This repository contains tools for processing and transcribing historical magazine pages and documents, separated based on explicit and nonexplicit content. The project offers two distinct workflows:

1. **Tesseract + AI Correction**: Uses Tesseract OCR for initial text extraction from grayscale images, then employs OpenAI's GPT-4o-mini to correct OCR errors
2. **Vision API Transcription**: Uses OpenAI's GPT-4o or GPT-5 Vision API to directly transcribe text from color images

### Project Structure

```
OCRAIProject/
├── explicit_OCR.py              # Process images with explicit content
├── nonexplicit_OCR.py           # Process images without explicit content
|-- pdf_to_images.py             # Process images from PDF to PNG - both greyscale and color, output to correct folder
├── tokenest.py                  # Estimate token usage for API calls
├── processed_imgs/              # Color images for processing
├── processed_imgs_gs/           # Grayscale images for processing
├── results_explicit/            # Output for explicit content
│   ├── tess_correction/        # Tesseract + AI corrected transcriptions
│   └── vision/                 # Vision API transcriptions
├── results_nonexplicit/        # Output for non-explicit content
│   ├── tess_correction/        # Tesseract + AI corrected transcriptions
│   └── vision/                 # Vision API transcriptions
└── requirements.txt            # Required packages for use of this directory
```

## 🔒 Copyright Notice

**Important**: This repository does NOT include the original PDF files or processed images due to copyright restrictions. The historical documents being processed are from women's liberation movement publications and are protected by copyright.

To use this code, you must:
- Obtain your own images legally
- Place them in the appropriate directories
- Ensure you have rights to process the materials

## 📦 Requirements

### System Requirements

- Python 3.8 or higher
- Tesseract OCR (for Tesseract + AI correction method)
- macOS, Linux, or Windows

### Python Packages

Install the required packages using pip:

```bash
pip install openai pytesseract pillow
```

Or use the requirements file:

```bash
pip install -r requirements.txt
```

#### Package Details:

- **openai** (>=1.0.0): OpenAI Python client for GPT models and Vision API
- **pytesseract** (>=0.3.10): Python wrapper for Tesseract OCR
- **Pillow** (>=10.0.0): Python Imaging Library for image processing

### Installing Tesseract OCR

#### macOS (using Homebrew):
```bash
brew install tesseract
```

#### Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr
```

#### Windows:
Download the installer from: https://github.com/UB-Mannheim/tesseract/wiki

## 🔑 Setup

### 1. Set OpenAI API Key

You must have an OpenAI API key with access to GPT-4o or GPT-5 models.

Set your API key as an environment variable:

**macOS/Linux:**
```bash
export OPENAI_API_KEY='your-api-key-here'
```

**Windows (PowerShell):**
```powershell
$env:OPENAI_API_KEY='your-api-key-here'
```

**Permanent setup (add to ~/.bashrc, ~/.zshrc, or equivalent):**
```bash
echo 'export OPENAI_API_KEY="your-api-key-here"' >> ~/.bashrc
source ~/.bashrc
```

### 2. Convert PDFs to Images

Use the `pdf_to_images.py` script to extract and process pages from PDF documents:

```bash
python pdf_to_images.py
```

**How it works:**
1. The script prompts you to select a PDF file from your system
2. All pages are extracted to a temporary directory
3. You're shown a preview of all pages with page numbers
4. You can select which specific pages to process (or all pages)
5. Selected pages are automatically converted to:
   - **Color version** → saved to `processed_imgs/`
   - **Grayscale version** → saved to `processed_imgs_gs/`
6. Temporary files are cleaned up after processing

**Interactive workflow:**
```
Select PDF pages to process:
0. Process ALL pages
1. Page 1
2. Page 2
3. Page 3

Enter page number(s) (comma-separated, or 0 for all): 1,3
```

**Output:**
```
✅ Saved color image: processed_imgs/document_page_001.png
✅ Saved grayscale image: processed_imgs_gs/document_page_001.png
✅ Saved color image: processed_imgs/document_page_003.png
✅ Saved grayscale image: processed_imgs_gs/document_page_003.png
```

### 3. Prepare Your Images

If you already have images (not from PDFs), place them directly in the appropriate directories:

- **Color images**: `processed_imgs/`
- **Grayscale images**: `processed_imgs_gs/`

Supported formats: PNG, JPG, JPEG

### 4. Configure Image Lists

Edit the image lists in `explicit_OCR.py` or `nonexplicit_OCR.py`:

```python
# Define the images for processing
grayscale_images = ["image1.png", "image2.png"]  # Your grayscale images
color_images = ["image1.png", "image2.png"]      # Your color images
```

## 🚀 Usage

### Running the Scripts

#### For Non-Explicit Content:
```bash
python nonexplicit_OCR.py
```

#### For Explicit Content:
```bash
python explicit_OCR.py
```

### Interactive Workflow

When you run either script, you'll be prompted to:

1. **Choose Processing Method:**
   ```
   === OCR Processing Options ===
   1. Tesseract + AI Correction (for grayscale images)
   2. OpenAI Vision API (for color images)
   3. Both methods
   
   Enter your choice (1, 2, or 3):
   ```

2. **Select Images to Process:**
   ```
   === Select Images to Process ===
   0. Process ALL images
   1. image1.png
   2. image2.png
   3. image3.png
   
   Enter image number(s) (comma-separated, or 0 for all):
   ```

### Processing Methods

#### Method 1: Tesseract + AI Correction

**How it works:**
1. Tesseract OCR extracts raw text from grayscale images
2. GPT-4o-mini corrects OCR errors while preserving original meaning
3. Results are saved to `results_*/tess_correction/`

**Best for:**
- Grayscale or black-and-white documents
- Clear, high-contrast text
- Lower API costs

**Output example:**
```
results_nonexplicit/tess_correction/image1_corrected.txt
```

#### Method 2: Vision API Transcription

**How it works:**
1. Images are resized to 1024x1024 pixels and converted to JPEG (quality=95)
2. Images are encoded to base64 format
3. GPT-4o or GPT-5 Vision API transcribes the text directly
4. Results are saved to `results_*/vision/`

**Best for:**
- Color images with complex layouts
- Images with mixed text and graphics
- Documents where layout preservation is critical

**Output example:**
```
results_nonexplicit/vision/image1_vision.txt
```

### Token Usage Estimation

To estimate token usage before processing:

```bash
python tokenest.py
```

This helps you:
- Predict API costs
- Identify images that may exceed token limits
- Optimize image sizes

## 💰 Cost Estimation

The scripts provide real-time cost estimates based on OpenAI's pricing:

```
🔹 Token Usage: Prompt = 15000, Completion = 2000, Total = 17000
💰 Estimated Cost: $0.095000
```

**Note**: Pricing may change. Check [OpenAI's pricing page](https://openai.com/pricing) for current rates.

## 🎯 Features

### Image Processing
- Automatic image resizing (max 1024x1024 pixels)
- PNG to JPEG conversion for reduced file size
- Grayscale conversion for Tesseract processing
- Quality preservation (JPEG quality=95)

### Transcription Quality
- Strict instructions to minimize AI hallucination
- Temperature=0.0 for consistent results
- Preserves original spelling, layout, and structure
- Marks illegible text clearly

### Error Handling
- Robust error catching and reporting
- Automatic file path resolution
- Missing file detection
- Detailed error messages with traceback

### Flexibility
- Process individual images or batches
- Choose processing method per session
- Separate workflows for different content types
- Customizable prompts and parameters

## 📊 Output Format

All transcriptions are saved as plain text files (.txt) with UTF-8 encoding.

**File naming convention:**
- Tesseract + AI: `{original_filename}_corrected.txt`
- Vision API: `{original_filename}_vision.txt`

## 🛠️ Customization

### Adjusting Image Quality

In `explicit_OCR.py` or `nonexplicit_OCR.py`, modify the `resize_and_convert_to_jpeg` function:

```python
img.save(jpeg_path, "JPEG", quality=95)  # Adjust quality (1-100)
```

### Changing Model

To use a different OpenAI model:

```python
# In transcribe_with_vision_api function
response = client.chat.completions.create(
    model="gpt-4o",  # Change to "gpt-5" or other available models
    ...
)
```

### Modifying Transcription Instructions

Edit the prompt in the `transcribe_with_vision_api` function to change how the AI transcribes text.

### Adjusting Token Limits

```python
max_tokens=4000,  # Increase if transcriptions are being cut off
```

## 🐛 Troubleshooting

### "API key not found" Error
- Ensure your OPENAI_API_KEY environment variable is set
- Restart your terminal after setting the variable
- Verify the key is valid

### "Could not find image" Error
- Check that images exist in the correct directories
- Verify file extensions match (case-sensitive on some systems)
- Use the `resolve_image_path` function for automatic extension detection

### "Token limit exceeded" Error
- Resize images to smaller dimensions
- Increase JPEG compression (lower quality value)
- Split large images into sections
- Use Tesseract method for very large documents

### Tesseract Not Found
- Verify Tesseract is installed: `tesseract --version`
- Add Tesseract to your PATH
- On macOS: `brew reinstall tesseract`

### Poor OCR Quality
- Increase image resolution before processing
- Adjust JPEG quality to 95-100
- Use grayscale images for Tesseract
- Ensure images have good contrast

## 📝 Best Practices

1. **Start with a few test images** to verify setup and estimate costs
2. **Use grayscale images** for Tesseract method when possible (lower cost)
3. **Batch process similar documents** together for efficiency
4. **Review transcriptions** for accuracy, especially with historical documents
5. **Keep original images** separate from processed versions
6. **Monitor token usage** to control costs
7. **Use quality=95** for JPEG conversion to balance file size and accuracy

## 🔄 Git Workflow

The `.gitignore` file excludes:
- PDF files (`*.pdf`)
- Image files (`*.png`, `*.jpg`, `*.jpeg`)
- Processed images directories
- Result directories
- Python cache files
- API keys and environment files

**To contribute:**
1. Fork the repository
2. Create a feature branch
3. Make your changes (code only, no images/PDFs)
4. Submit a pull request

## 📄 License

This code is provided for educational and research purposes. Users are responsible for ensuring they have rights to process any documents they use with this tool.

## 📧 Contact

For questions or issues, please open a GitHub issue or email at are4@clemson.edu

---

**Note**: This project is for educational and research purposes. Always respect copyright and obtain proper permissions before processing historical documents.
```

This README provides:
- Clear project overview and structure
- Complete setup instructions
- Detailed usage guide
- Troubleshooting section
- Best practices
- Copyright and legal considerations
- Information about the `.gitignore` for sensitive materials

The README is formatted for GitHub and includes emojis for better readability. It's comprehensive enough for someone to understand and reproduce your work while respecting copyright restrictions.
This README provides:
- Clear project overview and structure
- Complete setup instructions
- Detailed usage guide
- Troubleshooting section
- Best practices
- Copyright and legal considerations
- Information about the `.gitignore` for sensitive materials

The README is formatted for GitHub and includes emojis for better readability. It's comprehensive enough for someone to understand and reproduce your work while respecting copyright restrictions.
```