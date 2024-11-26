import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler

def setup_logger(name: str, log_file: str = None, level=logging.INFO):
    """Configure logger with console and file handlers"""
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Create file handler if log_file specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10485760,  # 10MB
            backupCount=5
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger