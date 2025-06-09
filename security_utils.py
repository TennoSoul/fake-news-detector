"""Security utilities for the fake news detection system."""

import hashlib
import html
import logging
import re
from typing import Dict, Optional
from urllib.parse import urlparse
import os
from pathlib import Path

class SecurityUtils:
    """Utility class for security functions."""
    
    @staticmethod
    def sanitize_input(text: str) -> str:
        """Sanitize input text.
        
        Args:
            text: Input text to sanitize
            
        Returns:
            Sanitized text
        """
        # Remove potential HTML/JS
        text = html.escape(text)
        
        # Remove potentially dangerous characters
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        
        return text
        
    @staticmethod
    def validate_url(url: str) -> bool:
        """Validate URL format and safety.
        
        Args:
            url: URL to validate
            
        Returns:
            bool: True if URL is valid and safe
        """
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception:
            return False
            
    @staticmethod
    def verify_file_integrity(filepath: str, expected_hash: Optional[str] = None) -> Dict[str, str]:
        """Verify file integrity using SHA-256.
        
        Args:
            filepath: Path to file
            expected_hash: Expected SHA-256 hash
            
        Returns:
            Dict with file hash and validation status
        """
        try:
            with open(filepath, 'rb') as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()
                
            result = {
                'file': Path(filepath).name,
                'hash': file_hash,
                'valid': True if not expected_hash else file_hash == expected_hash
            }
            
            return result
        except Exception as e:
            logging.error(f"File integrity check failed: {e}")
            return {
                'file': Path(filepath).name,
                'hash': None,
                'valid': False,
                'error': str(e)
            }
            
    @staticmethod
    def validate_request_data(request_data: Dict) -> bool:
        """Validate API request data.
        
        Args:
            request_data: Request data to validate
            
        Returns:
            bool: True if request data is valid
        """
        # Check if text field exists and is string
        if not isinstance(request_data.get('text'), str):
            return False
            
        # Check text length
        if len(request_data.get('text', '')) > 10000:  # Maximum text length
            return False
            
        return True
        
class ModelSecurity:
    """Security functions for ML models."""
    
    @staticmethod
    def verify_model_integrity(model_dir: str) -> Dict[str, Dict]:
        """Verify integrity of model files.
        
        Args:
            model_dir: Directory containing model files
            
        Returns:
            Dict with integrity check results for each file
        """
        try:
            results = {}
            model_files = ['config.json', 'model.safetensors', 'vocab.txt']
            
            for file in model_files:
                filepath = os.path.join(model_dir, file)
                if os.path.exists(filepath):
                    results[file] = SecurityUtils.verify_file_integrity(filepath)
                else:
                    results[file] = {
                        'file': file,
                        'hash': None,
                        'valid': False,
                        'error': 'File not found'
                    }
                    
            return results
        except Exception as e:
            logging.error(f"Model integrity verification failed: {e}")
            return {}
            
    @staticmethod
    def validate_model_input(text: str, max_length: int = 512) -> bool:
        """Validate model input.
        
        Args:
            text: Input text to validate
            max_length: Maximum allowed text length
            
        Returns:
            bool: True if input is valid
        """
        # Check input type
        if not isinstance(text, str):
            return False
            
        # Check input length
        if len(text) > max_length:
            return False
            
        # Check for potentially malicious content
        if any(pattern in text.lower() for pattern in ['<script>', 'javascript:', 'data:']):
            return False
            
        return True
        
def setup_secure_logging():
    """Configure secure logging settings."""
    return {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'verbose': {
                'format': '{levelname} {asctime} {module} {process:d} {thread:d} {message}',
                'style': '{',
            },
        },
        'handlers': {
            'file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'filename': 'logs/system.log',
                'maxBytes': 1024 * 1024 * 5,  # 5 MB
                'backupCount': 5,
                'formatter': 'verbose',
            },
        },
        'loggers': {
            'fake_news_detection': {
                'handlers': ['file'],
                'level': 'INFO',
                'propagate': True,
            },
        },
    }
