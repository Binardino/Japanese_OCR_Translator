"""
Database engine and session factory for the Japanese OCR Translator.

Import SessionLocal wherever you need a DB session:
    db = SessionLocal()
    try:
        db.add(obj)
        db.commit()
    finally:
        db.close()

Tables are created automatically on first import via Base.metadata.create_all().
"""
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from database.models import Base

DATABASE_URL = "sqlite:///./output/japanese_translation.db"

engine = create_engine(DATABASE_URL, 
                    connect_args={'check_same_thread': False})

SessionLocal = sessionmaker(bind=engine, 
                            autocommit=False, 
                            autoflush=False)

Base.metadata.create_all(bind=engine) # == to CREATE TABLE IF NOT EXISTS
