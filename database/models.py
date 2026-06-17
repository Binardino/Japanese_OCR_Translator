"""
SQLAlchemy ORM models for the Japanese OCR Translator dictionary.

Two tables:
- Translations : one row per processed screenshot (source sentence + translation)
- Vocabulary   : one row per vocabulary word extracted from that screenshot

Relationship: Translations 1 → N Vocabulary (one sentence yields several words).
Navigate with: translation.vocabulary (list of words) / word.sentence (parent sentence).
"""
from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class Translations(Base):
    """One processed screenshot — stores the source Japanese sentence and its English translation."""

    __tablename__ = "translations"

    id          = Column(Integer, primary_key=True, index=True)
    game_name   = Column(String(100), nullable=False, index=True)   # derived from the parent folder name
    filename    = Column(String(100), nullable=False, index=True)   # original image filename
    jap_raw     = Column(Text, nullable=False)                      # Japanese sentence as seen (kanji)
    jap_kana    = Column(Text, nullable=False)                      # same sentence 100% in kana
    translation = Column(Text, nullable=False)                      # English translation
    created_at  = Column(DateTime, default=lambda: datetime.now(timezone.utc))  # UTC, set at insert time

    # One-to-many: a sentence has several vocabulary entries
    vocabulary  = relationship("Vocabulary", back_populates="sentence")


class Vocabulary(Base):
    """One vocabulary word extracted from a translated screenshot."""

    __tablename__ = "vocabulary"

    id             = Column(Integer, primary_key=True, index=True)
    translation_id = Column(Integer, ForeignKey("translations.id"), nullable=False, index=True)  # FK → parent sentence
    word           = Column(Text, nullable=False)
    reading        = Column(Text, nullable=False)   # hiragana reading of the word
    meaning        = Column(Text, nullable=False)   # English definition
    jlpt           = Column(String(3), index=True)  # "N1"–"N5", nullable (model may not always know)

    # Many-to-one: navigate back to the source sentence
    sentence = relationship("Translations", back_populates="vocabulary")
