FROM python:3.10-slim

# Install Tesseract and Japanese language model
RUN apt-get update && \
    apt-get install -y tesseract-ocr tesseract-ocr-jpn libglib2.0-0 libsm6 libxrender1 libxext6 poppler-utils && \
    apt-get clean

# Set up workdir
WORKDIR /app
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Default command
CMD ["python", "main.py"]
