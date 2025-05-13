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

def draw_text_panel(original_img, text_lines):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_color = (0, 0, 0)
    line_height = 30

    width = original_img.shape[1]
    height = max(original_img.shape[0], line_height * (len(text_lines) + 1))
    panel = np.ones((height, width, 3), dtype=np.uint8) * 255

    y0 = 30
    for i, line in enumerate(text_lines):
        cv2.putText(panel, line, (10, y0 + i * line_height), font, font_scale, font_color, 1, cv2.LINE_AA)

    return panel

def process_image(filename):
    input_path = os.path.join(INPUT_DIR, filename)
    base_name = os.path.splitext(filename)[0]

    cropped, processed = preprocess_image(input_path)
    data = pytesseract.image_to_string(processed, config=tess_config).strip()

    lines = [line for line in data.split("\n") if line.strip()]
    results = []

    for line in lines:
        try:
            translation = GoogleTranslator(source='ja', target='en').translate(line)
            results.append(f"{line}\n→ {translation}")
        except Exception as e:
            results.append(f"{line}\n→ [Translation Error: {e}]")

    # Save text output
    with open(os.path.join(OUTPUT_DIR, f"{base_name}.txt"), "w", encoding="utf-8") as f:
        for r in results:
            f.write(r + "\n\n")

    # Generate side-by-side output image
    text_panel = draw_text_panel(cropped, results)
    output_image = np.hstack((cropped, text_panel))
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base_name}_translated.jpg"), output_image)


if __name__ == "__main__":
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    for fname in os.listdir(INPUT_FOLDER):
        fpath = os.path.join(INPUT_FOLDER, fname)
        img = preprocess_image(fpath)
        jp_text = extract_text(img)    
        en_text = translator.translate(jp_text)
        draw_translation(fpath, en_text)
        print(f"[✓] Processed {fname}")
        