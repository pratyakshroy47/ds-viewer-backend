# Dataset Viewer Backend

A FastAPI-based backend service for viewing and managing datasets and audio files.

## Setup

1. Clone the repository:

```
git clone <repository-url>
cd ds-viewer-backend
```
2. Run the setup script:

```
chmod +x setup.sh
./setup.sh
```

3. Configure your environment:
- Copy `.env.example` to `.env` and update the values
- Add your Google Cloud credentials file (storage.json)

4. Start the server:
```
uvicorn app.main:app --reload
```


## Features

- Dataset loading and pagination
- Data filtering and quality checks
- Audio file caching and streaming
- Google Cloud Storage integration
- Efficient caching system

