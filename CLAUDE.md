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

**Tech stack:**
- `manga-ocr` — manga/game-specialized OCR (replaces Tesseract)
- `google-generativeai` — Gemini API for translation (replaces Google Translate)
- `watchdog` — real-time monitoring of the `input/` folder
- `opencv-python` + `Pillow` — image preprocessing and rendering
- `fugashi` + `jamdict` — Japanese tokenization and vocabulary dictionary
- Docker — containerization

**Workflow:**
```
input/ → watchdog → manga-ocr → Gemini API → output/ (.txt + .jpg)
                                            + word_dictionary.json
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
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── ROADMAP.md            # Learning roadmap
├── README.md
├── input/                # Screenshots to process (not committed)
└── output/               # Results (not committed)
```

---

## Development Phases (see ROADMAP.md)

- **Phase 0** — Project cleanup
- **Phase 1** — manga-ocr (replaces Tesseract)
- **Phase 2** — Gemini API (replaces Google Translate)
- **Phase 3** — Vocabulary dictionary (Jamdict, already coded but disabled)
- **Phase 4** — watchdog (automation)
- **Phase 5** — Final Docker + README

---

## Important Technical Notes

- `FONT_PATH` is required in `.env` (NotoSansCJK for Japanese character rendering)
- `GEMINI_API_KEY` is required in `.env`
- The crop (12%-70% height, 18%-82% width) is calibrated for PS Vita — adjust if needed
- `MangaOcr()` must be instantiated ONCE at module level (heavy model ~400MB)
- `update_word_dictionary()` is in `main.py` but commented out — to be enabled in Phase 3
- The Jamdict DB path (`/root/.jamdict/jamdict.db`) must be verified after `python3 -m jamdict import`
