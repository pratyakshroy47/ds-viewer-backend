#!/bin/bash

# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Windows, use: .\venv\Scripts\activate

# Install requirements
pip install -r requirements.txt

# Create necessary directories
mkdir -p logs
mkdir -p cache/datasets
mkdir -p cache/audio

# Create empty .env file if it doesn't exist
if [ ! -f .env ]; then
    cp .env.example .env
fi

echo "Setup complete! Don't forget to:"
echo "1. Configure your .env file"
echo "2. Add your Google Cloud credentials (storage.json)"