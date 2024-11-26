from fastapi.testclient import TestClient
import pytest
from app.main import app

client = TestClient(app)

def test_load_dataset(test_dataset_path):
    response = client.post(
        "/api/v1/dataset/load",
        json={"file_path": test_dataset_path}
    )
    assert response.status_code == 200
    data = response.json()
    assert "total_rows" in data
    assert "columns" in data
    assert "sample_data" in data

def test_get_audio():
    response = client.get(
        "/api/v1/dataset/audio/test",
        params={"path": "gs://test-bucket/test.wav"}
    )
    assert response.status_code == 400  # Should fail without GCP credentials