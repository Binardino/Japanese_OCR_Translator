import os
import cv2
import preprocessing
from google import genai
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from config import INPUT_DIR, OUTPUT_DIR, FONT_PATH, DICT_PATH, JAMDICT_DB, GEMINI_API_KEY
import time
#jam = Jamdict(JAMDICT_DB)

# The client gets the API key from the environment variable `GEMINI_API_KEY`.
client = genai.Client(api_key=GEMINI_API_KEY)

gemini_translate_prompt = """1. The Role
                You have to translate the below input text - from Japanese - to English.
                Preserve the tone of JRPG dialogue. 
                Character names, spell names and item names should stay in their original form if untranslatable.

                2. The Context
                This current API call is part of a Python data pipeline. 
                The user is a Japanese learner, using video games in Japanese to learn & practice Japanse. 
                Extract dialogues and scenes from video game screenshots (from Dsi / New 3DS games or from PS Vita / PSP games).
                The input image is a raw photo of the videogame console made by phone. 
                Identify the console screen and target the text contents in the game.

                3. Expected Output:
                TRANSLATION:
                Return only the translated text, no explanation, no commentary.

                VOCABULARY
                List the 3 - 5 most noteworthy vocabulary words or interesting grammar expression to remember from this extract
                - <word in kanji> (<reading>) : <meaning in English>"""

# Paths
os.makedirs(OUTPUT_DIR, exist_ok=True)


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

def process_image(filename):
    input_path = os.path.join(INPUT_DIR, filename)
    base_name = os.path.splitext(filename)[0]

    image = Image.open(input_path)
    results = []

    try:
        translation = client.models.generate_content(
                model="gemini-2.0-flash", 
                contents=[image, gemini_translate_prompt]
                )
        print(translation.text)

        results.append(f"{translation.text}")
    except Exception as e:
        results.append(f"[Translation Error: {e}]")

    time.sleep(7)
    # Generate output image with only cropped area
    text_panel = draw_text_panel(image, results)
    cropped_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)  # PIL is RGB, cv2.imwrite expects BGR

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
