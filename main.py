import os
import cv2
import pytesseract
from PIL import Image, ImageDraw, ImageFont
from deep_translator import GoogleTranslator
import numpy as np
from config import INPUT_DIR, OUTPUT_DIR, FONT_PATH, DICT_PATH, JAMDICT_DB
from fugashi import Tagger
from jamdict import Jamdict

#jam = Jamdict(JAMDICT_DB)

# Paths
INPUT_DIR  = "input"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# OCR config
tess_config = "--psm 6 -l jpn"

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    # Crop to screen area (adjust values depending on your setup)
    h, w, _ = image.shape
    cropped = image[int(h*0.12):int(h*0.70), int(w*0.18):int(w*0.82)]

    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    denoised = cv2.fastNlMeansDenoising(thresh, h=10)
    return cropped, denoised

def draw_text_panel(original_img, text_lines, font_path=FONT_PATH):
    """
    Create a white panel with translated text printed next to the image.

    Args:
        original_img (numpy.ndarray): The original image to match the height with.
        text_lines (list): List of strings, each line is a Japanese sentence + its translation (alternating).
        font_path (str): Path to the Japanese-capable TrueType font (ttc or ttf).

    Returns:
        numpy.ndarray: Panel image with text written, ready to be concatenated with the original image.
    """

    # Font settings
    font_size = 20
    line_spacing = 10
    font = ImageFont.truetype(font_path, font_size)

    # Estimate height
    line_height = font_size + line_spacing
    panel_height = max(original_img.shape[0], line_height * len(text_lines) * 2)
    panel_width = 500

    # Create blank white image
    panel = Image.new("RGB", (panel_width, panel_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(panel)

    y = 10
    for line in text_lines:
        if '→' in line:
            jp, en = line.split('→', 1)
            jp = jp.strip()
            en = en.strip()
            draw.text((10, y), jp, font=font, fill=(0, 0, 0))
            y += line_height
            draw.text((10, y), f"→ {en}", font=font, fill=(100, 100, 100))
            y += line_height + 5
        else:
            draw.text((10, y), line.strip(), font=font, fill=(0, 0, 0))
            y += line_height + 5

    # Convert back to OpenCV format
    panel_np = np.array(panel)
    return panel_np


def process_image(filename):
    input_path = os.path.join(INPUT_DIR, filename)
    base_name = os.path.splitext(filename)[0]

    cropped, processed = preprocess_image(input_path)
    data = pytesseract.image_to_string(processed, config=tess_config).strip()

    lines = [line for line in data.split("\n") if line.strip()]
    results = []
    japanese_sentences = []

    for line in lines:
        try:
            translation = GoogleTranslator(source='ja', target='en').translate(line)
            results.append(f"{line}\n→ {translation}")
            japanese_sentences.append(line)
        except Exception as e:
            results.append(f"{line}\n→ [Translation Error: {e}]")

    # Save text output
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, f"{base_name}.txt"), "w", encoding="utf-8") as f:
        for r in results:
            f.write(r + "\n\n")

    # Generate output image with only cropped area
    text_panel = draw_text_panel(cropped, results)
    output_image = np.hstack((cropped, text_panel))
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base_name}_translated.jpg"), output_image)

    # Update consolidated dictionary
    #update_word_dictionary(japanese_sentences)


def main():
    for filename in os.listdir(INPUT_DIR):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"Processing: {filename}")
            process_image(filename)


if __name__ == "__main__":
    main()
