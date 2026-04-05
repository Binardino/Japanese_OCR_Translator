import os
import cv2
from manga_ocr import MangaOcr
from google import genai
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from config import INPUT_DIR, OUTPUT_DIR, FONT_PATH, DICT_PATH, JAMDICT_DB, GEMINI_API_KEY
from fugashi import Tagger
from jamdict import Jamdict

#jam = Jamdict(JAMDICT_DB)
# The client gets the API key from the environment variable `GEMINI_API_KEY`.
client = genai.Client(api_key=GEMINI_API_KEY)

# Paths
os.makedirs(OUTPUT_DIR, exist_ok=True)

# OCR config
mocr = MangaOcr()

def preprocess_image(image_path):
    """
    Load the image and crop to the manga panel area.
    With PIL & MangaOcr, adjust the cropping values based on your specific manga layout.
    
    Input : Path to the manga page image.
    Output: Cropped image focused on the manga panel area.
    """
    image = Image.open(image_path)
    # Crop to screen area (adjust values depending on your setup)
    #PIL outputs : (width, height)
    w, h = image.size
    #PIL crop box: (left, upper, right, lower)
    cropped = image.crop((int(w*0.18), int(h*0.12), int(w*0.82), int(h*0.70)))

    return cropped

def draw_text_panel(original_img, text_lines, font_path=FONT_PATH):
    """
    Create a white panel with translated text printed next to the image.

    Args:
        original_img (numpy.ndarray): The original image to match the height with.
        text_lines (list): List of strings, each line is a Japanese sentence + its translation (alternating).
        font_path (str): Path to the Japanese-capable TrueType font (ttc or ttf).

    Returns:
        PIL Image: Panel image with text written, ready to be concatenated with the original image.
    """

    # Font settings
    font_size = 20
    line_spacing = 10
    font = ImageFont.truetype(font_path, font_size)

    # Estimate height
    line_height = font_size + line_spacing
    panel_height = max(original_img.size[1], line_height * len(text_lines) * 2)
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

"""
def update_word_dictionary(japanese_sentences? jap_dict=JAMDICT_DB):
    if os.path.exists(DICT_PATH):
        with open(DICT_PATH, "r", encoding="utf-8") as f:
            word_dict = json.load(f)
    else:
        word_dict = {}

    tagger = Tagger()
    jam = Jamdict(jap_dict)
    for sentence in japanese_sentences:
        for word in tagger(sentence):
            surface = word.surface
            if surface in word_dict:
                continue
            entry = jam.lookup(surface)
            if entry.entries:
                first_entry = entry.entries[0]
                meanings = first_entry.sense_summary()
                readings = [r.text for r in first_entry.reading_elements]
                onyomi = [r for r in readings if "音" in r or "オン" in r]
                kunyomi = [r for r in readings if "訓" in r or "くん" in r]
                word_dict[surface] = {
                    "meanings": meanings,
                    "readings": readings,
                    "kunyomi": kunyomi,
                    "onyomi": onyomi
                }

    with open(DICT_PATH, "w", encoding="utf-8") as f:
        json.dump(word_dict, f, ensure_ascii=False, indent=2)
"""

def process_image(filename):
    input_path = os.path.join(INPUT_DIR, filename)
    base_name = os.path.splitext(filename)[0]

    cropped = preprocess_image(input_path)
    data    = mocr(cropped)
    results = []

    try:
        translation = client.models.generate_content(
                model="gemini-2.0-flash", 
                contents=f"Translate from Japanese in English the value {data}"
                )
        print(translation.text)

        results.append(f"{data}\n→ {translation.text}")
    except Exception as e:
        results.append(f"{data}\n→ [Translation Error: {e}]")

    # Generate output image with only cropped area
    text_panel = draw_text_panel(cropped, results)
    #cropped is numpy array in RGB format, convert to BGR for OpenCV
    cropped_np = cv2.cvtColor(np.array(cropped), cv2.COLOR_RGB2BGR)

    output_image = np.hstack((cropped_np, text_panel))
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base_name}_translated.jpg"), output_image)

    return results

def main(existing_files=None):
    if existing_files is None:
        existing_files = set()

    all_results = {}
    for filename in os.listdir(INPUT_DIR):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            if filename in existing_files:
                print(f"Skipping (already processed): {filename}")
                continue
            print(f"Processing: {filename}")
            all_results[filename] = process_image(filename)

    # Save text output
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "output.md"), "a", encoding="utf-8") as f:
        for key, value in all_results.items():
            f.write(f"### {key}\n")
            for trad in value:
                trad = trad.replace("\n", " ")
                f.write(f"- {trad}\n")
            f.write("\n")

if __name__ == "__main__":
    existing_files = set()
    output_md = os.path.join(OUTPUT_DIR, "output.md")
    if os.path.exists(output_md):
        with open(output_md, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("### "):
                    existing_files.add(line.strip()[4:])
    main(existing_files)
