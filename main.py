import os
import cv2
import pytesseract
from PIL import Image, ImageDraw, ImageFont
from deep_translator import GoogleTranslator
import numpy as np

# Paths
INPUT_DIR = "input"
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


if __name__ == "__main__":
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    for fname in os.listdir(INPUT_FOLDER):
        fpath = os.path.join(INPUT_FOLDER, fname)
        img = preprocess_image(fpath)
        jp_text = extract_text(img)    
        en_text = translator.translate(jp_text)
        draw_translation(fpath, en_text)
        print(f"[✓] Processed {fname}")
        