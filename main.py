import os
import cv2
import pytesseract
from PIL import Image, ImageDraw, ImageFont
from deep_translator import GoogleTranslator

INPUT_FOLDER = "input"
OUTPUT_FOLDER = "output"
LANG = "jpn"

translator = GoogleTranslator(source='ja', target='en')

def preprocess_image(path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    return resized

def extract_text(image):
    config = '--oem 3 --psm 6 -l jpn'
    return pytesseract.image_to_string(image, config=config)

def draw_translation(original_path, translation):
    base = Image.open(original_path).convert("RGB")
    W, H = base.size
    font = ImageFont.load_default()

    # Create new image to hold original + translated
    combined = Image.new("RGB", (W * 2, H), (255, 255, 255))
    combined.paste(base, (0, 0))

    draw = ImageDraw.Draw(combined)
    draw.text((W + 10, 10), translation, fill=(0, 0, 0), font=font)

    output_path = os.path.join(OUTPUT_FOLDER, os.path.basename(original_path))
    combined.save(output_path)

if __name__ == "__main__":
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    for fname in os.listdir(INPUT_FOLDER):
        fpath = os.path.join(INPUT_FOLDER, fname)
        img = preprocess_image(fpath)
        jp_text = extract_text(img)    
        en_text = translator.translate(jp_text)
        draw_translation(fpath, en_text)
        print(f"[✓] Processed {fname}")
        