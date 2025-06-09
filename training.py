"""Training utilities for fake news detection models."""

import logging
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any

def prepare_data(texts: List[str], labels: List[int], test_size: float = 0.2, 
                val_size: float = 0.2, random_state: int = 42) -> Dict[str, Tuple[List[str], List[int]]]:
    """Split data into train/val/test sets."""    
    try:
        # First split into train+val and test
        train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
            texts, labels,
            test_size=test_size,
            random_state=random_state,
            stratify=labels
        )
        
        # Then split train into train and val
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            train_val_texts, train_val_labels,
            test_size=val_size,
            random_state=random_state,
            stratify=train_val_labels
        )
        
        return {
            'train': (train_texts, train_labels),
            'val': (val_texts, val_labels),
            'test': (test_texts, test_labels)
        }
        
    except ValueError as e:
        logging.warning(f"Could not perform stratified split: {e}. Falling back to random split.")
        train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
            texts, labels,
            test_size=test_size,
            random_state=random_state
        )
        
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            train_val_texts, train_val_labels,
            test_size=val_size,
            random_state=random_state
        )
        
        return {
            'train': (train_texts, train_labels),
            'val': (val_texts, val_labels),
            'test': (test_texts, test_labels)
        }

def train_model(model, train_data: Tuple[List[str], List[int]], 
                val_data: Optional[Tuple[List[str], List[int]]] = None, 
                **kwargs) -> Dict[str, List[float]]:
    """Train a model using the optimized training implementation.
    
    Args:
        model: The model to train (should implement train() method)
        train_data: Tuple of (texts, labels) for training
        val_data: Optional tuple of (texts, labels) for validation
        **kwargs: Additional arguments to pass to model.train()
        
    Returns:
        Training history dictionary
    """
    train_texts, train_labels = train_data
    if val_data:
        val_texts, val_labels = val_data
    else:
        val_texts = val_labels = None

    # Extract training parameters that should not be passed to train()
    epochs = kwargs.pop('epochs', None)  # Remove epochs from kwargs if present
    
    # Use the model's optimized training implementation
    return model.train(
        train_texts=train_texts,
        train_labels=train_labels,
        val_texts=val_texts,
        val_labels=val_labels
    )

def evaluate_model(model, test_data: Tuple[List[str], List[int]], 
                  save_path: Optional[str] = None) -> Dict[str, Any]:
    """Evaluate a trained model.
    
    Args:
        model: Trained model instance
        test_data: Tuple of (texts, labels) for testing
        save_path: Optional path to save evaluation results
        
    Returns:
        Dictionary containing evaluation metrics
    """
    import os
    import numpy as np
    from sklearn.metrics import classification_report, confusion_matrix
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    test_texts, test_labels = test_data
    
    # Get model predictions
    results = model.evaluate(test_texts, test_labels)
    pred_labels = results.get('predictions', [])
    
    # Calculate metrics
    metrics = {
        'accuracy': results['metrics']['accuracy'],
        'classification_report': classification_report(test_labels, pred_labels),
    }
    
    # Create and save confusion matrix plot if save_path provided
    if save_path:
        cm = confusion_matrix(test_labels, pred_labels)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(save_path, 'confusion_matrix.png'))
        plt.close()
        
        # Save metrics to text file
        with open(os.path.join(save_path, 'metrics.txt'), 'w') as f:
            f.write(f"Accuracy: {metrics['accuracy']:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(metrics['classification_report'])
    
    return {
        'metrics': metrics,
        'predictions': pred_labels
    }

def plot_training_history(history: Dict[str, List[float]], save_path: Optional[str] = None):
    """Plot training history metrics.
    
    Args:
        history: Dictionary containing training metrics
        save_path: Optional path to save plots
    """
    import os
    
    metrics = ['loss', 'accuracy'] if 'accuracy' in history else ['loss']
    
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        
        # Plot training metric
        if f'train_{metric}' in history:
            plt.plot(history[f'train_{metric}'], label=f'Training {metric}')
            
        # Plot validation metric
        if f'val_{metric}' in history:
            plt.plot(history[f'val_{metric}'], label=f'Validation {metric}')
            
        plt.title(f'Model {metric}')
        plt.xlabel('Epoch')
        plt.ylabel(metric.capitalize())
        plt.legend()
        
        if save_path:
            plt.savefig(os.path.join(save_path, f'{metric}_history.png'))
        plt.close()
