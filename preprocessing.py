import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class TextPreprocessor(BaseEstimator, TransformerMixin):
    """Text preprocessing transformer compatible with sklearn Pipeline.
    
    Handles:
    - Text cleaning (HTML, URLs, special chars)
    - Tokenization
    - Stop word removal
    - Lemmatization
    """
    
    def __init__(self, remove_stopwords=True, lemmatize=True):
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.lemmatizer = WordNetLemmatizer() if lemmatize else None
        self.stop_words = set(stopwords.words('english')) if remove_stopwords else set()

    def fit(self, X, y=None):
        """Required for sklearn Pipeline compatibility."""
        return self
        
    def transform(self, X):
        """Transform a list of texts."""
        return [self.preprocess(text) for text in X]
        
    def preprocess(self, text):
        """Clean and preprocess text for analysis."""
        if not isinstance(text, str):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Remove punctuation but keep sentence structure
        text = re.sub(r'[^\w\s.,!?]', '', text)
        
        # Remove repeated whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenization
        tokens = word_tokenize(text)
        
        # Remove stop words if enabled
        if self.remove_stopwords:
            tokens = [token for token in tokens if token not in self.stop_words]
        
        # Lemmatization if enabled  
        if self.lemmatize:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        # Join tokens back into text
        return ' '.join(tokens)

    def prepare_for_bert(self, text, max_length=512):
        """Prepare text specifically for BERT model."""
        # Basic cleaning while preserving sentence structure
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Truncate if longer than max_length
        words = text.split()
        if len(words) > max_length:
            text = ' '.join(words[:max_length])
        
        return text
