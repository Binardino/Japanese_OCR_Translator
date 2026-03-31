# ROADMAP — Japanese OCR Translator

Learning roadmap for building a Japanese OCR pipeline for PS Vita / 3DS game screenshots.

---

## Target Architecture

```
input/ (PS Vita/3DS screenshots)
    ↓ watchdog (real-time detection)
manga-ocr (manga/game-specialized OCR → Japanese text)
    ↓
Gemini API (JA→EN translation, "video game" system prompt)
    ↓
output/ (JP+EN .txt pairs + composite .jpg)
    + word_dictionary.json (Jamdict — personal vocabulary)
```

---

## Phase 0 — Cleanup

**Goal:** Start with a clean foundation.

**Key concept:** Config/code separation — why use `.env` + `config.py` instead of hardcoded paths?
> Answer: paths change depending on the environment (local vs Docker vs another machine). `.env` externalizes config without touching the code.

**Tasks:**
- [ ] Delete `main_old.py`, `main_old2.py`, `Dockerfile_old`
- [ ] Delete empty placeholder folders: `data/`, `ocr/`, `project/`, `translator/`
- [ ] Create `.gitignore` — ignore `input/`, `output/`, `.env`, `__pycache__/`, `*.pyc`
- [ ] Create `.claudeignore` — exclude screenshots from Claude's context
- [ ] Create `.env.example` with all variables (no secret values)
- [ ] Fix `main.py` lines 14-15: delete the two lines that overwrite the imported config

---

## Phase 1 — Replace Tesseract with manga-ocr

**Why?**
Tesseract is a general-purpose OCR engine trained on document text. It struggles with:
- Low resolutions (3DS screens: 400×240px)
- Stylized pixel fonts from games
- Vertical Japanese text
- Furigana

`manga-ocr` ([kha-white/manga-ocr-base](https://huggingface.co/kha-white/manga-ocr-base)) is a Transformer model trained specifically on manga and game text — far more accurate for this use case.

**Concepts to learn:**
1. **Loading a HuggingFace model**: `from manga_ocr import MangaOcr; mocr = MangaOcr()`
2. **numpy vs PIL images**: OpenCV works with numpy arrays (BGR), manga-ocr expects a PIL image (RGB). Conversion: `Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))`
3. **Module-level instantiation**: the model is ~400MB and takes ~5s to load. Putting it inside the loop = performance disaster.

**Changes in `main.py`:**
- Replace `pytesseract` import with `from manga_ocr import MangaOcr`
- Instantiate `mocr = MangaOcr()` once at module level
- In `process_image()`: convert the cropped image to PIL and call `mocr(image_pil)`
- Remove `tess_config`

**Changes in `requirements.txt`:**
- Replace `pytesseract` with `manga-ocr`

**Test:** Process one image, print the extracted Japanese text to the console.

---

## Phase 2 — Replace Google Translate with Gemini

**Why?**
Google Translate translates line by line with no context. A Japanese RPG has specific terms (spell names, items, characters) that an LLM understands better with a good system prompt.

**Concepts to learn:**
1. **System prompt**: a persistent instruction that frames the model's behavior for the entire session. Example: *"You are an expert Japanese video game translator."*
2. **Gemini API** (`google-generativeai`): `genai.configure(api_key=...)` then `model.generate_content(text)`
3. **API key security**: never hardcoded in source, always in `.env` → loaded via `config.py`
4. **Error handling**: wrap the API call in `try/except` (rate limit, timeout)

**Changes in `config.py`:**
- Add `GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")`
- Raise a `ValueError` if missing

**Changes in `main.py`:**
- Replace `from deep_translator import GoogleTranslator` with `import google.generativeai as genai`
- Import `GEMINI_API_KEY` from config
- Configure and instantiate the Gemini model at module level (not in the loop)
- Replace `GoogleTranslator(...).translate(line)` with `model.generate_content(line).text`

**Changes in `requirements.txt`:**
- Replace `deep-translator` with `google-generativeai`

**Test:** Compare a game sentence translated by Google Translate vs Gemini.

---

## Phase 3 — Re-enable the Vocabulary Dictionary

**Why?**
Learning Japanese through games means building vocabulary in context. The `update_word_dictionary()` function in `main.py` is fully coded but disabled (wrapped in a `"""..."""` block comment).

**Concepts to learn:**
1. **Morphological tokenization**: Japanese has no spaces → can't split words naively. Fugashi (Python wrapper for MeCab) analyzes grammatical structure to extract individual words.
2. **Jamdict**: JMdict database (Japanese-English dictionary). `jam.lookup(word)` returns definitions, readings, onyomi, kunyomi.
3. **JSON persistence**: `json.load()` to read, `json.dump()` to write. `ensure_ascii=False` to preserve Japanese characters.

**Changes in `main.py`:**
- Add `import json` at the top
- Fix line 78: `japanese_sentences?` → `japanese_sentences,` (syntax bug)
- Uncomment the `"""..."""` block (lines 77-108) to restore the function
- Uncomment the call on line 141: `#update_word_dictionary(japanese_sentences)`

**Prerequisite:** Jamdict must be initialized. Run once:
```bash
python3 -m jamdict import
```

**Test:** After processing an image, check that `output/word_dictionary.json` contains entries with readings and meanings.

> **Note:** If `Jamdict(JAMDICT_DB)` fails, try `Jamdict()` with no argument — the library finds its data automatically.

---

## Phase 4 — Automation with watchdog

**Why?**
Current workflow: take a screenshot → copy it to `input/` → run `python main.py` → check `output/`. With watchdog, the script runs in the background and reacts as soon as a file appears.

**Concepts to learn:**
1. **Event-driven programming**: instead of periodically checking ("polling"), you subscribe to an event and react when it fires.
2. **watchdog**: `Observer` watches a folder, `FileSystemEventHandler` defines what to do on each event (`on_created`, `on_modified`, etc.)
3. **Main loop**: `while True: time.sleep(1)` + `KeyboardInterrupt` → `observer.stop()` + `observer.join()` for clean shutdown.

**Changes in `main.py`:**
- Add `import time` and watchdog imports
- Rework `main()`: first process existing files in `input/`, then start the observer
- Create a `ScreenshotHandler(FileSystemEventHandler)` class with `on_created()`
- Handle clean shutdown on `KeyboardInterrupt`

**Changes in `requirements.txt`:**
- Add `watchdog`

**Test:** Run `python main.py`, drop a screenshot into `input/`, verify it is processed automatically.

---

## Phase 5 — Final Docker + Documentation

**Why?**
Docker ensures reproducibility: same result on any machine, without manually installing Tesseract, fonts, etc.

**Tasks:**
- [ ] Update `Dockerfile`:
  - Remove `tesseract-ocr` and `tesseract-ocr-jpn` (no longer needed)
  - Add manga-ocr pre-download: `RUN python -c "from manga_ocr import MangaOcr; MangaOcr()"`
  - Uncomment `RUN python3 -m jamdict import`
- [ ] Verify `docker-compose.yml` passes `.env` to the container
- [ ] Rewrite `README.md`: Docker setup, local setup, env variables, crop tuning note

---

## Recommended Order

```
Phase 0 → Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5
  (15min)   (1-2h)    (1h)      (1h)      (1h)      (30min)
```

Each phase is independently testable. Start with Phase 1 — it is the core of the pipeline.

---

## Dependency Summary

| Package | Usage | Phase |
|---|---|---|
| `manga-ocr` | Specialized Japanese OCR | 1 |
| `opencv-python` | Image preprocessing (crop, grayscale) | 0 |
| `Pillow` | Image + text rendering (draw_text_panel) | 0 |
| `google-generativeai` | Translation via Gemini API | 2 |
| `python-dotenv` | `.env` loading | 0 |
| `fugashi` | Japanese morphological tokenization | 3 |
| `jamdict` | JMdict dictionary | 3 |
| `numpy` | Image array manipulation | 0 |
| `watchdog` | input/ folder monitoring | 4 |
