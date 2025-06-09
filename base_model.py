from abc import ABC, abstractmethod
from typing import Optional, Dict, List, Any, Union
import torch
import torch.cuda
import logging

# Initialize CUDA if available
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.init()
    logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    logging.info("CUDA not available, using CPU")

class BaseModel(ABC):
    """Abstract base class for fake news detection models"""
    
    @abstractmethod
    def predict(self, text: str, device: Optional[torch.device] = None) -> dict:
        """Predict if text is fake news
        
        Args:
            text: Input text to analyze
            device: Optional device to run prediction on
        Returns:
            Dictionary with prediction results
        """
        pass
    
    @abstractmethod
    def batch_predict(self, texts: List[str], device: Optional[torch.device] = None) -> List[Dict[str, Any]]:
        """Predict for multiple texts
        
        Args:
            texts: List of input texts
            device: Optional device to run prediction on
        Returns:
            List of prediction dictionaries
        """
        pass
    
    @abstractmethod
    def evaluate(self, texts: List[str], labels: List[Union[int, float]], 
                device: Optional[torch.device] = None) -> Dict[str, float]:
        """Evaluate model performance
        
        Args:
            texts: List of input texts
            labels: List of true labels 
            device: Optional device to run evaluation on
        Returns:
            Dictionary with evaluation metrics
        """
        pass
        
    @abstractmethod
    def to(self, device: torch.device) -> 'BaseModel':
        """Move model to specified device
        
        Args:
            device: Device to move model to
        Returns:
            Self for method chaining
        """
        pass
