import os
import cv2
import preprocessing
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from config import INPUT_DIR, OUTPUT_DIR, FONT_PATH, MODEL
import ollama
from io import BytesIO
from pathlib import Path
import json
from database.models import Translations, Vocabulary
from database.database import SessionLocal

# Load prompt once at startup — kept in a separate file so prompt edits don't require touching Python code
LLM_translate_prompt = (Path(__file__).parent / "prompts" / "translate.txt").read_text()

# JSON schema passed to Ollama's structured output feature — enforces exact key names at the sampling level,
# so the model physically cannot produce wrong keys regardless of how it reasons internally.
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "japanese_raw":  {"type": "string"},
        "japanese_kana": {"type": "string"},
        "translation":   {"type": "string"},
        "vocabulary": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "word":    {"type": "string"},
                    "reading": {"type": "string"},
                    "meaning": {"type": "string"},
                    "jlpt":    {"type": "string"}
                },
                "required": ["word", "reading", "meaning"]
            }
        },
        "source": {"type": "string"}
    },
    "required": ["japanese_raw", "japanese_kana", "translation", "vocabulary"]
}

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

def process_image(image_path, game_name):
    """
    Run the full pipeline on a single image: preprocess → VL model to translate.

    Args:
        image_path (str): Full path to the image file.
        game_name (str): Game title injected into the prompt (derived from parent folder name).

    Returns:
        dict | None: Parsed JSON dict from the model (keys: japanese_raw, japanese_kana,
                     translation, vocabulary, source), or None on error.
    """
    base_name = Path(image_path).stem  # filename without extension, used for the output JPG name

    image  = preprocessing.preprocess(image_path)
    # Inject the game name into the prompt — replaces the {game_source} placeholder in translate.txt
    prompt = LLM_translate_prompt.replace("{game_source}", game_name)

    # Ollama expects image bytes, not a file path — encode the PIL image to JPEG in memory
    buffer = BytesIO()
    image.save(buffer, format="JPEG")

    data = None      # will hold the parsed JSON dict on success
    text_lines = []  # display lines for the side-by-side panel

    try:
        response = ollama.chat(
                model=MODEL,
                messages=[{"role"   : "user",
                        "content": prompt,
                        "images" : [buffer.getvalue()]
                        }],
                format=RESPONSE_SCHEMA,  # structured output: enforces exact key names at sampling level
                options={"think": False} #, "num_predict": 2048}
                )
        raw_content = response["message"]["content"]
        print(f"[DEBUG] raw_content length: {len(raw_content)}, image bytes: {len(buffer.getvalue())}")
        print(f"[CONTENT] {raw_content}")

        # Strip markdown code fences the model may wrap around its JSON output
        clean_str = raw_content.replace('\n', ' ').strip()
        if clean_str.startswith("```json"):
            clean_str = clean_str[7:]
        elif clean_str.startswith("```"):
            clean_str = clean_str[3:]
        if clean_str.endswith("```"):
            clean_str = clean_str[:-3]
        clean_str = clean_str.strip()

        data = json.loads(clean_str)
        # Normalize top-level keys: "Japanese raw" → "japanese_raw", "Translation" → "translation", etc.
        data = {k.lower().replace(' ', '_'): v for k, v in data.items()}

        text_lines = [f"{data.get('japanese_raw', '?')} → {data.get('translation', '?')}"]

        translation = Translations(game_name = game_name,
                                filename     = Path(image_path).name,
                                jap_raw      = data.get('japanese_raw', ''),
                                jap_kana     = data.get('japanese_kana') or data.get('kana', ''),
                                translation  = data.get('translation', ''))

        for word in data.get('vocabulary', []):  # each word is a dict with word/reading/meaning/jlpt keys
            vocab = Vocabulary(word     = word.get('word', ''),
                                reading = word.get('reading', ''),
                                meaning = word.get('meaning', ''),
                                jlpt    = word.get('jlpt') or word.get('jlpt_level') or word.get('level')
                            )

            translation.vocabulary.append(vocab)

        db = SessionLocal()
        try:
            db.add(translation)  # cascade → vocabulary too with relation
            db.commit()
        finally:
            db.close()

    except Exception as e:
        print(f"[FULL ERROR]: {e}")
        text_lines = [f"[Translation Error: {e}]"]
        data = None  # don't return a partial/malformed dict as if it were a success

    # Build side-by-side output: preprocessed screen on the left, translation panel on the right
    text_panel = draw_text_panel(image, text_lines)
    cropped_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)  # PIL RGB → OpenCV BGR for imwrite
    output_image = np.hstack((cropped_np, text_panel))
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base_name}_translated.jpg"), output_image)

    return data

def main(existing_files=None):
    """
    Process all images in INPUT_DIR (recursively), skipping already-processed files.

    Each subfolder of INPUT_DIR is treated as a game title. The relative key
    "GameName/filename.jpg" is used to uniquely identify images across games
    and to match against entries in output.md for skip detection.

    Args:
        existing_files (set): Relative keys already written to output.md. Default: empty set.
    """
    if existing_files is None:
        existing_files = set()

    all_results = {}
    for image_path in Path(INPUT_DIR).rglob("*"):
        if image_path.suffix.lower() in ('.png', '.jpg', '.jpeg'):
            game_name    = image_path.parent.name            # subfolder name = game title
            relative_key = f"{game_name}/{image_path.name}" # unique key across all games

            if relative_key in existing_files:
                print(f"Skipping (already processed): {relative_key}")
                continue

            print(f"Processing: {relative_key}")
            all_results[relative_key] = process_image(str(image_path), game_name)

    # Append to output.md — "a" mode never erases results from previous runs
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "output.md"), "a", encoding="utf-8") as f:
        for key, value in all_results.items():
            if value is None:
                continue
            f.write(f"### {key}\n")
            f.write(f"- {value.get('japanese_raw', '?')} → {value.get('translation', '?')}\n")
            f.write("\n")

if __name__ == "__main__":
    # Build the skip set from output.md before running — lines like "### GameName/img1.jpg"
    # are parsed to extract "GameName/img1.jpg" and stored in existing_files
    existing_files = set()
    output_md = os.path.join(OUTPUT_DIR, "output.md")
    if os.path.exists(output_md):
        with open(output_md, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("### "):
                    existing_files.add(line.strip()[4:])  # strip "### " prefix → "GameName/img1.jpg"
    main(existing_files)
