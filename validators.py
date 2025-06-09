from urllib.parse import urlparse
import re
import os

def validate_url(url: str) -> bool:
    """Validate if string is a proper URL."""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False

def validate_twitter_credentials(credentials: dict) -> bool:
    """Validate Twitter API credentials."""
    required_keys = {'consumer_key', 'consumer_secret', 'access_token', 'access_token_secret'}
    return all(key in credentials for key in required_keys)

def validate_content(content: str) -> bool:
    """Validate if content is meaningful."""
    if not isinstance(content, str):
        return False
    # Remove whitespace and check length
    cleaned = re.sub(r'\s+', '', content)
    return len(cleaned) >= 50

def validate_bert_input(text: str, max_length: int = 512) -> bool:
    """Validate if text is suitable for BERT processing."""
    if not isinstance(text, str):
        return False
    words = text.split()
    return 0 < len(words) <= max_length

def validate_model_config(config: dict) -> bool:
    """Validate model configuration."""
    required_keys = {
        'model_name', 'max_length', 'batch_size', 
        'learning_rate', 'epochs'
    }
    return all(key in config for key in required_keys)

def validate_output_path(path: str) -> bool:
    """Validate if path is writable."""
    try:
        from pathlib import Path
        output_dir = Path(path)
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir.is_dir() and os.access(path, os.W_OK)
    except Exception:
        return False
