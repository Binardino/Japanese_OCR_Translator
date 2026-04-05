# CLAUDE.md — Japanese OCR Translator

## Assistant Role

This is a learning project. The user wants to **learn to develop** this OCR pipeline themselves, not receive ready-made code.

**Claude should:**
- Explain concepts (the *why*, not just the *what*)
- Guide toward the right approach when the user is stuck
- Review and correct code written by the user
- Suggest directions, not complete solutions

**Claude should NOT:**
- Write entire functions on behalf of the user
- Refactor code that wasn't asked about
- Add unrequested features

---

## Project Context

Python OCR pipeline to translate screenshots from Japanese PS Vita / 3DS games.
The user is learning Japanese by playing Japanese games and wants to automate screenshot translation.

**Input reality:** photos taken with a smartphone of a physical 3DS XL screen (not native screenshots). Images are 3000x4000px (Samsung S25 Ultra). Quality is inconsistent — different angles, scan lines, varying positions.

**Tech stack:**
- `manga-ocr` — manga/game-specialized OCR ✓
- `google-genai` — Gemini 2.0 Flash for translation (replaced deep-translator) ✓
- `watchdog` — real-time monitoring of the `input/` folder
- `opencv-python` + `Pillow` — image preprocessing and rendering ✓
- `fugashi` + `jamdict` — Japanese tokenization and vocabulary dictionary (Phase 3, disabled)
- `poetry` — dependency management ✓
- Docker — containerization (Phase 5)

**Workflow:**
```
data/ → manga-ocr (OCR) → Gemini 2.0 Flash (translate + vocabulary) → output/
                                                                       ├── output.md
                                                                       └── *_translated.jpg
```

---

## Project Structure

```
Japanese_OCR_Translator/
├── .env                  # Local variables (not committed)
├── .env.example          # Template (committed)
├── .gitignore
├── .claudeignore
├── config.py             # Loads env vars
├── main.py               # Main pipeline
├── pyproject.toml        # Poetry deps (no requirements.txt)
├── poetry.lock
├── Dockerfile
├── docker-compose.yml
├── ROADMAP.md
├── README.md
├── LEARNINGS.md          # Technical concepts learned during development
├── data/                 # Input photos (not committed)
└── output/               # Results (not committed)
```

---

## Development Phases

- **Phase 0** — Project cleanup ✓
- **Phase 1** — manga-ocr (replaces Tesseract) ✓
- **Phase 2** — Gemini API (replaces Google Translate) — IN PROGRESS
  - Gemini client initialized ✓
  - Single API call per image (not per line) ✓
  - Rate limiting: time.sleep(7) between calls ✓
  - Skip already-processed files via output.md headers ✓
  - Prompt with role + context + structured output ✓
  - `return results` bug fix pending ✓ — verify before next run
- **Phase 3** — Vocabulary dictionary (Jamdict, already coded but disabled)
- **Phase 4** — watchdog (automation)
- **Phase 5** — Final Docker + README

---

## Important Technical Notes

### Config
- `FONT_PATH` required in `.env` (NotoSansCJK: `/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc`)
- `GEMINI_API_KEY` required in `.env`
- `INPUT_DIR` defaults to `./data` (changed from `./input`)

### OCR
- `MangaOcr()` instantiated ONCE at module level as `mocr` (~400MB, ~5s load)
- manga-ocr returns a **single string** per image (not newline-separated lines)
- Crop (18%-82% width, 12%-70% height) calibrated for PS Vita — needs adjustment for 3DS phone photos
- For phone photos of 3DS: text box is at the BOTTOM of the screen → crop should target bottom 30%
- Debug: add `cropped.save("debug_crop.jpg")` in `preprocess_image()` to visualize crop

### Gemini API
- SDK: `from google import genai` / `client = genai.Client(api_key=GEMINI_API_KEY)`
- Model: `gemini-2.0-flash`
- Free tier: 15 req/min → `time.sleep(7)` between calls
- 429 RESOURCE_EXHAUSTED = rate limit hit, not quota exhausted
- Prompt structure: Role → Context → Expected Output (TRANSLATION + VOCABULARY sections)
- `translation.text` = response string (not `translation` which is the Response object)

### Image pipeline
- PIL throughout, convert to numpy BGR only for `cv2.imwrite()` at the end
- `cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)` for the conversion

### Output
- `output.md` opened in `"a"` (append) mode — never `"w"` (would erase previous results)
- Skip logic: read `output.md` headers (`### filename`) before running → build `existing_files` set
- `results` is a list of strings → iterate to write markdown bullets

### Dependency management
- Poetry only (`pyproject.toml` + `poetry.lock`), Python 3.11
- `opencv-python` installs as `cv2` (package name ≠ import name)
- `deep-translator` removed, replaced by `google-genai`

### GPU / WSL2
- WSL2 GPU passthrough: Windows driver handles it, no NVIDIA driver inside WSL
- RTX 5080 Laptop = Blackwell architecture (sm_120) → requires PyTorch **nightly** (not stable)
- Install: `poetry run pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128 --force-reinstall`
- Verify: `poetry run python -c "import torch; print(torch.cuda.is_available())"`
- manga-ocr uses GPU automatically when available

### Known limitations
- Phone photos of physical console screens are poor OCR input (scan lines, angles, variable crop)
- For a production-quality pipeline, native screenshots (emulator/capture card) would be needed
- Future direction: Gemini Vision (send image directly) would handle phone photos better

### Git / commit style
- Conventional commits: `feat(phase2):`, `chore:`, `docs:` etc.
- No `Co-Authored-By` footer
- All documentation in English
