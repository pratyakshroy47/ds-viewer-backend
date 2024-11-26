import os
import sys
import json
import logging
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.services.dataset_service import DatasetService
from app.utils.logger import setup_logger
from fastapi.testclient import TestClient
from app.main import app

# Setup test logger
logger = setup_logger("system_health_test", "logs/system_test.log")

class SystemHealthCheck:
    def __init__(self):
        self.client = TestClient(app)
        self.dataset_service = DatasetService()

    def check_file_exists(self, file_path: str) -> bool:
        """Verify if the file exists and is accessible"""
        try:
            path = Path(file_path)
            exists = path.exists()
            logger.info(f"File existence check: {file_path} - {'Found' if exists else 'Not Found'}")
            return exists
        except Exception as e:
            logger.error(f"Error checking file existence: {file_path}", exc_info=True)
            return False

    def check_file_permissions(self, file_path: str) -> bool:
        """Verify if we have read permissions for the file"""
        try:
            path = Path(file_path)
            readable = os.access(path, os.R_OK)
            logger.info(f"File permissions check: {file_path} - {'Readable' if readable else 'Not Readable'}")
            return readable
        except Exception as e:
            logger.error(f"Error checking file permissions: {file_path}", exc_info=True)
            return False

    def check_file_format(self, file_path: str) -> bool:
        """Verify if the file format is supported"""
        supported_formats = ['.csv', '.json', '.jsonl', '.parquet']
        try:
            extension = Path(file_path).suffix.lower()
            is_supported = extension in supported_formats
            logger.info(f"File format check: {extension} - {'Supported' if is_supported else 'Not Supported'}")
            return is_supported
        except Exception as e:
            logger.error(f"Error checking file format: {file_path}", exc_info=True)
            return False

    def validate_jsonl_content(self, file_path: str) -> dict:
        """Validate JSONL file content structure"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # Read first line for structure validation
                first_line = f.readline().strip()
                sample_record = json.loads(first_line)
                
                # Count total lines
                f.seek(0)
                total_lines = sum(1 for line in f if line.strip())
                
            result = {
                "status": "valid",
                "total_records": total_lines,
                "sample_keys": list(sample_record.keys()),
                "message": "File structure is valid"
            }
            logger.info(f"Content validation successful: {file_path}")
            return result
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON format in file: {file_path}", exc_info=True)
            return {"status": "invalid", "message": f"Invalid JSON format: {str(e)}"}
        except Exception as e:
            logger.error(f"Error validating file content: {file_path}", exc_info=True)
            return {"status": "error", "message": str(e)}

    def test_api_endpoint(self, file_path: str) -> dict:
        """Test the dataset loading API endpoint"""
        try:
            response = self.client.post(
                "/api/v1/dataset/load",
                json={"file_path": file_path}
            )
            logger.info(f"API endpoint test: {response.status_code}")
            return {
                "status_code": response.status_code,
                "response": response.json() if response.status_code == 200 else None
            }
        except Exception as e:
            logger.error(f"Error testing API endpoint", exc_info=True)
            return {"status_code": 500, "error": str(e)}

    def run_health_check(self, file_path: str) -> dict:
        """Run all health checks and return results"""
        logger.info(f"Starting system health check for: {file_path}")
        
        results = {
            "file_path": file_path,
            "checks": {
                "file_exists": self.check_file_exists(file_path),
                "file_permissions": self.check_file_permissions(file_path),
                "file_format": self.check_file_format(file_path),
            }
        }
        
        # Only proceed with content validation if basic checks pass
        if all(results["checks"].values()):
            results["content_validation"] = self.validate_jsonl_content(file_path)
            results["api_test"] = self.test_api_endpoint(file_path)
        else:
            logger.warning("Basic checks failed, skipping content validation and API test")
            results["status"] = "failed"
            results["message"] = "Basic file checks failed"
            
        logger.info("Health check completed")
        return results

def main():
    """Main function to run the health check"""
    # Specify your file path here
    file_path = "/Users/temp/projects/ds-viewer-backend/data/bible.jsonl"
    
    # Enable verbose logging
    logger.setLevel(logging.DEBUG)
    
    health_checker = SystemHealthCheck()
    results = health_checker.run_health_check(file_path)
    
    print("\nSystem Health Check Results:")
    print("===========================")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()