# Technical Learnings — Japanese OCR Translator

Concepts and tools learned during development of this project.

---

## Python Fundamentals

### String immutability
```python
s = "hello world"
s.replace("hello", "hi")  # ← result discarded, s is unchanged
s = s.replace("hello", "hi")  # ← correct: assign the result
```
All Python string methods return a new string. The original is never modified.

### Dict iteration
```python
d = {"a": 1, "b": 2}
for key in d:           # iterates keys only → "a", "b"
for key, val in d:      # TypeError: cannot unpack string
for key, val in d.items():  # correct → ("a", 1), ("b", 2)
```

### File modes
| Mode | Behavior |
|---|---|
| `"r"` | Read only |
| `"w"` | Write — **erases existing content** |
| `"a"` | Append — adds to end, never erases |

Choosing `"w"` when you need persistence across runs is a common bug.

### Sets vs lists for lookup
```python
existing = ["a", "b", "c"]  # O(n) lookup
"a" in existing              # scans entire list

existing = {"a", "b", "c"}  # O(1) lookup (hash table)
"a" in existing              # instant
```
Use `set()` when you need to check membership frequently.

### Module-level instantiation
```python
# BAD — loads 400MB model on every function call
def process_image(filename):
    mocr = MangaOcr()   # ← called 54 times = 54 loads
    ...

# GOOD — loads once at startup
mocr = MangaOcr()       # ← module level

def process_image(filename):
    data = mocr(image)  # ← reuses loaded model
```

### Default mutable parameter trap
```python
def main(existing_files=None):       # correct
    if existing_files is None:
        existing_files = set()

def main(existing_files=set()):      # BUG — shared across all calls
```

### f-strings do not execute arbitrary code
```python
value = [1, 2, 3]
f"result: [i*2 for i in {value}]"  # literal text, not a list comprehension
f"result: {[i*2 for i in value]}"  # correct — expression inside {}
```

### return placement
```python
def process():
    result = compute()
    return result         # ← everything after this is unreachable
    # Update dictionary   # ← never executed
    update()
```

---

## Separation of Concerns

Each function should do one thing:
- `preprocess_image()` — loads and crops the image, returns PIL Image
- `process_image()` — orchestrates OCR + translation for one file, returns results
- `main()` — iterates files, accumulates results, writes output

`main()` is the **orchestrator**. Processing logic belongs in dedicated functions.

---

## Image Processing

### PIL coordinate system
```python
image.size          # (width, height) — width first
image.crop((left, upper, right, lower))  # box = (x1, y1, x2, y2)
```

### RGB vs BGR
- **PIL** works in **RGB**
- **OpenCV** works in **BGR**

Always convert at the boundary:
```python
# PIL → OpenCV (for cv2.imwrite)
img_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

# OpenCV → PIL
img_pil = Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))
```

### Crop calibration
Fixed percentage crops are device-specific. For phone photos of a physical screen:
- The screen position varies with each photo
- Debug: save the crop to inspect it (`cropped.save("debug.jpg")`)
- For 3DS dialogue boxes: text is in the **bottom 30%** of the screen

---

## OpenCV ≠ OCR — Critical Distinction

**OpenCV** (`cv2`) is a **computer vision** library. It processes images but does NOT read text.

**OCR engines** (manga-ocr, EasyOCR, Tesseract) actually **read text** from images.

```
Photo
  ↓ OpenCV: grayscale, blur, threshold, deskew, contour detection
Prepared image
  ↓ OCR engine: reads the text
Raw text string
  ↓ LLM / translation API
Translation
```

OpenCV transforms:
| Function | What it does |
|---|---|
| `cv2.cvtColor(..., cv2.COLOR_BGR2GRAY)` | Convert to grayscale |
| `cv2.GaussianBlur()` | Reduce noise |
| `cv2.threshold()` with Otsu | Auto-binarize (black/white) |
| `cv2.findContours()` | Detect text regions |
| `cv2.getPerspectiveTransform()` | Correct perspective distortion |
| `cv2.imwrite()` | Save image to disk |

---

## Dependency Management

### Package name ≠ import name
| Install (pip/poetry) | Import in Python |
|---|---|
| `opencv-python` | `import cv2` |
| `Pillow` | `from PIL import Image` |
| `google-genai` | `from google import genai` |

### Poetry workflow
```bash
poetry install          # install all deps from pyproject.toml
poetry run python main.py  # run in virtual environment
poetry run pip install X   # install extra package in venv
```

---

## APIs

### Gemini API (google-genai SDK)
```python
from google import genai
client = genai.Client(api_key=GEMINI_API_KEY)  # module level

response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents="your prompt"
)
text = response.text   # ← .text, not the response object itself
```

### Rate limiting
- Free tier: 15 requests/minute
- 429 RESOURCE_EXHAUSTED = rate limit, not quota exhausted
- Fix: `time.sleep(7)` after each API call (outside try/except)

### Prompt engineering
Good prompt structure:
1. **Role** — who Gemini should be
2. **Context** — what the data is and where it comes from
3. **Expected output format** — exact structure to return

Structured output example:
```
TRANSLATION:
<translated text>

VOCABULARY:
- word (reading) : meaning
```

### API key security
- Never hardcode in source files
- Store in `.env` (not committed)
- Load via `os.getenv()` in `config.py`
- Validate at startup: `raise ValueError(...)` if missing

---

## GPU / CUDA / WSL2

### Architecture
```
RTX GPU (hardware)
    ↓
NVIDIA driver — installed on Windows ONLY
    ↓ (WSL2 passthrough via /dev/dxg)
CUDA Toolkit — installed inside WSL (no driver)
    ↓
PyTorch — bundles its own CUDA runtime in wheels
```

### Key rules
- Never install NVIDIA driver inside WSL
- `nvidia-smi` in WSL = GPU passthrough is working
- PyTorch CUDA wheels bundle their own runtime → full CUDA toolkit often unnecessary

### RTX 5080 (Blackwell, sm_120)
```bash
# Stable PyTorch doesn't support sm_120 yet → use nightly
poetry run pip install --pre torch torchvision \
  --index-url https://download.pytorch.org/whl/nightly/cu128 \
  --force-reinstall

# Verify
poetry run python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

### CUDA version vs architecture
| GPU generation | Architecture | CUDA | PyTorch |
|---|---|---|---|
| RTX 30xx (Ampere) | sm_86 | 11.x | stable |
| RTX 40xx (Ada) | sm_89 | 12.x | stable |
| RTX 50xx (Blackwell) | sm_120 | 12.8 | nightly |

---

## Pipeline Design Patterns

### Idempotency — skip already-processed files
```python
# Read previous results to know what was already done
existing_files = set()
if os.path.exists(output_md):
    with open(output_md, "r") as f:
        for line in f:
            if line.startswith("### "):
                existing_files.add(line.strip()[4:])  # strip "### "

# Skip during processing
if filename in existing_files:
    continue
```

### Accumulate then write (not write inside loop)
```python
# BAD — writes partial results, hard to structure
for file in files:
    result = process(file)
    write_to_output(result)   # one write per file

# GOOD — accumulate, then write structured output once
all_results = {}
for file in files:
    all_results[file] = process(file)
write_markdown(all_results)   # one structured write at the end
```

### Single API call per unit of work
```python
# BAD — 1 API call per line = N calls per image
for line in lines:
    translation = call_api(line)

# GOOD — 1 API call per image
translation = call_api(full_text)
```
Fewer API calls = fewer rate limit hits, better context for the model.

---

## LLM Output Robustness

### The key naming problem
LLMs often deviate from the exact key names specified in the prompt, even when their internal reasoning shows they know the correct schema. Examples encountered:

| Intended key | Model produced |
|---|---|
| `japanese_raw` | `"Japanese raw"` |
| `japanese_kana` | `"Kana"` |
| `translation` | `"Translation"` |
| `jlpt` | `"JLPT Level"`, `"level"`, `"jlpt_level"` |

Two complementary defenses:

**1. Ollama Structured Outputs** — enforces exact key names at the token sampling level, not as a prompt instruction. The model physically cannot produce a different key:
```python
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "japanese_raw": {"type": "string"},
        "translation":  {"type": "string"},
        ...
    },
    "required": ["japanese_raw", "translation", ...]
}
response = ollama.chat(model=MODEL, messages=[...], format=RESPONSE_SCHEMA)
```

**2. Python-side key normalization** — safety net in case the model still deviates:
```python
data = json.loads(raw_content)
data = {k.lower().replace(' ', '_'): v for k, v in data.items()}
# "Japanese raw" → "japanese_raw", "Translation" → "translation"
```

Always use both: structured output prevents the problem, normalization catches edge cases.

### `data = None` in the except block
When `json.loads()` succeeds but a subsequent key access fails, `data` already holds a wrong-keyed dict. Without resetting it, the caller receives a non-None value and crashes:
```python
data = None
try:
    data = json.loads(raw_content)   # succeeds, sets data
    _ = data['japanese_raw']          # KeyError → jumps to except
    ...
except Exception as e:
    data = None  # reset: don't leak a malformed dict as if it were a success
```

### Hallucination under structured outputs
When a model can't read the image content (poor quality, scan lines), and structured output forces it to fill every required field, it **fabricates content** rather than failing. This is a different failure mode than key naming errors:
- With free-form output: model might return prose, wrong keys, or refuse
- With structured output: model returns valid JSON with correct keys but **invented content**

The fix is not in the code — it's in the input data quality. For phone photos of physical screens, a VLM will hallucinate on any unreadable image. The `None` return from `process_image()` on exception prevents bad data from reaching the database.

### `think=False` with structured outputs (qwen3)
`options={"think": True}` and `format=<schema>` can conflict in Ollama with qwen3 models: thinking tokens interfere with the constrained sampling, causing the final JSON to diverge from what the model reasoned internally. Always set `think=False` when using structured output mode.
