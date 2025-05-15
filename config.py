# config.py
import os
from dotenv import load_dotenv

load_dotenv()

FONT_PATH  = os.getenv("FONT_PATH")
INPUT_DIR  = os.getenv("INPUT_DIR", "./input")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./output")
DICT_PATH  = os.getenv("DICT_PATH")
JAMDICT_DB = os.getenv("JAMDICT_DB")

if not FONT_PATH:
    raise ValueError("FONT_PATH is not defined in the .env file")