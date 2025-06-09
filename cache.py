import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

class Cache:
    """Simple cache for scraped content."""
    
    def __init__(self, cache_dir: str = "cache", expiry_hours: int = 24):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.expiry_delta = timedelta(hours=expiry_hours)
        
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get item from cache if not expired."""
        cache_file = self.cache_dir / f"{key}.json"
        
        if not cache_file.exists():
            return None
            
        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)
                cached_time = datetime.fromisoformat(data['cached_at'])
                
                if datetime.now() - cached_time > self.expiry_delta:
                    cache_file.unlink()
                    return None
                    
                return data['content']
        except Exception:
            return None
            
    def set(self, key: str, content: Dict[str, Any]):
        """Save item to cache."""
        cache_file = self.cache_dir / f"{key}.json"
        
        try:
            with open(cache_file, 'w') as f:
                json.dump({
                    'cached_at': datetime.now().isoformat(),
                    'content': content
                }, f)
        except Exception:
            pass
