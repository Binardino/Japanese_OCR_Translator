FROM python:3.10-slim

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Set up workdir
WORKDIR /app
COPY . /app

# Install system packages
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-jpn \
    poppler-utils \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1 \
    fonts-noto-cjk \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install requirements
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Create jamdict data directory
RUN mkdir -p /root/.jamdict/data

# Download and build the dictionary
#RUN python3 -m jamdict import

# Default command
CMD ["python", "main.py"]