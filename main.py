import os
import cv2
import preprocessing
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from config import INPUT_DIR, OUTPUT_DIR, FONT_PATH, MODEL
import ollama
from io import BytesIO
from pathlib import Path

# Load prompt once at startup — kept in a separate file so prompt edits don't require touching Python code
LLM_translate_prompt = (Path(__file__).parent / "prompts" / "translate.txt").read_text()

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
        list[str]: List containing the model's JSON response (or error message).
    """
    base_name = Path(image_path).stem  # filename without extension, used for the output JPG name

    image  = preprocessing.preprocess(image_path)
    # Inject the game name into the prompt — replaces the {game_source} placeholder in translate.txt
    prompt = LLM_translate_prompt.replace("{game_source}", game_name)

    # Ollama expects image bytes, not a file path — encode the PIL image to JPEG in memory
    buffer = BytesIO()
    image.save(buffer, format="JPEG")

    results = []

    try:
        response = ollama.chat(
                model=MODEL,
                messages=[{"role"   : "user",
                           "content": prompt,
                           "images" : [buffer.getvalue()]
                           }],
                options={"think": False}  # qwen3 thinking mode outputs to message["thinking"], leaving content empty
                )
        results.append(response["message"]["content"])

    except Exception as e:
        print(f"[FULL ERROR]: {e}")
        results.append(f"[Translation Error: {e}]")

    # Build side-by-side output: preprocessed screen on the left, translation panel on the right
    text_panel = draw_text_panel(image, results)
    cropped_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)  # PIL RGB → OpenCV BGR for imwrite
    output_image = np.hstack((cropped_np, text_panel))
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base_name}_translated.jpg"), output_image)

    return results

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
            f.write(f"### {key}\n")
            for trad in value:
                trad = trad.replace("\n", " ")
                f.write(f"- {trad}\n")
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
