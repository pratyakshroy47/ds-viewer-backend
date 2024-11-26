import pytest
from fastapi.testclient import TestClient
from app.main import app
import logging

@pytest.fixture(autouse=True)
def setup_logging():
    # Configure logging for tests
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def test_dataset_path(tmp_path):
    # Create a test dataset
    dataset_path = tmp_path / "test.csv"
    dataset_path.write_text("column1,column2\nvalue1,value2")
    return str(dataset_path)