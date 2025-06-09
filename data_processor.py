import re
import string
from typing import List, Dict, Any
import pandas as pd
from bs4 import BeautifulSoup
from textblob import TextBlob
import numpy as np
from collections import Counter
from torch.utils.data import Dataset

class DataProcessor:
    """Handles data preprocessing and cleaning for fake news detection."""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text content."""
        if not isinstance(text, str):
            return ""
        
        # Remove HTML
        text = BeautifulSoup(text, "html.parser").get_text()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Remove special characters but keep sentence structure
        text = re.sub(r'[^\w\s.,!?]', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    @staticmethod
    def prepare_batch(texts: List[str], max_length: int = 512) -> List[str]:
        """Prepare a batch of texts for model input."""
        return [
            DataProcessor.clean_text(text)[:max_length] 
            for text in texts if text
        ]

    @staticmethod
    def format_results(predictions: List[Dict[str, Any]]) -> pd.DataFrame:
        """Format prediction results into a DataFrame."""
        return pd.DataFrame(predictions).assign(
            timestamp=pd.Timestamp.now()
        )

    @staticmethod
    def get_text_stats(text: str) -> Dict[str, float]:
        """Calculate statistical features of text."""
        if not isinstance(text, str) or not text.strip():
            return {
                'word_count': 0,
                'avg_word_length': 0,
                'sentence_count': 0,
                'complexity_score': 0
            }
        
        words = text.split()
        sentences = [s for s in re.split(r'[.!?]+', text) if s.strip()]
        
        return {
            'word_count': len(words),
            'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
            'sentence_count': len(sentences),
            'complexity_score': len(set(words)) / len(words) if words else 0
        }

    @staticmethod
    def analyze_sentiment(text: str) -> Dict[str, float]:
        """Analyze sentiment of text using TextBlob."""
        if not isinstance(text, str) or not text.strip():
            return {'polarity': 0, 'subjectivity': 0}
        
        blob = TextBlob(text)
        return {
            'polarity': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity
        }

    def process_batch(self, texts: List[str], include_stats: bool = True) -> List[Dict[str, Any]]:
        """Process a batch of texts with enhanced features."""
        results = []
        for text in texts:
            if not text:
                continue
                
            cleaned = self.clean_text(text)
            result = {
                'original_length': len(text),
                'cleaned_length': len(cleaned),
                'cleaned_text': cleaned
            }
            
            if include_stats:
                result.update(self.get_text_stats(cleaned))
                result.update(self.analyze_sentiment(cleaned))
                
            results.append(result)
            
        return results

    @staticmethod
    def extract_keywords(text: str, top_n: int = 10) -> List[str]:
        """Extract important keywords from text."""
        if not isinstance(text, str) or not text.strip():
            return []
            
        # Remove common words and punctuation
        stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to'])
        words = re.findall(r'\b\w+\b', text.lower())
        words = [w for w in words if w not in stop_words and len(w) > 2]
        
        # Get most common words
        return [word for word, _ in Counter(words).most_common(top_n)]

class NewsDataset(Dataset):
    # Label mapping for LIAR dataset
    LABEL_MAP = {
        "TRUE": 0,
        "mostly-true": 1,
        "half-true": 2,
        "barely-true": 3,
        "FALSE": 4,
        "pants-fire": 5
    }
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Convert string labels to numeric
        self.labels = [self.LABEL_MAP[label.lower()] if isinstance(label, str) else label for label in labels]
    
    # ...existing code...
    
    @staticmethod
    def get_label_name(label_id):
        """Convert numeric label back to string label"""
        reverse_map = {v: k for k, v in NewsDataset.LABEL_MAP.items()}
        return reverse_map.get(label_id, "unknown")
