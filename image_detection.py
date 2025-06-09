"""Image detection module for detecting AI-generated images."""

import numpy as np
import cv2
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, Union
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import torch
from .noise_analysis import NoiseAnalyzer

class ImageDetector:
    def __init__(self, config):
        """Initialize image detector with configuration.
        
        Args:
            config: Configuration object containing media settings
        """
        self.config = config
        self.noise_analyzer = NoiseAnalyzer(config)
        self.model = self._load_model()
        self.device = torch.device('cuda' if torch.cuda.is_available() and 
                                 config.gpu_config['use_gpu'] else 'cpu')
                                 
    def _load_model(self) -> Union[RandomForestClassifier, LogisticRegression]:
        """Load the trained classifier model."""
        try:
            model_path = Path(self.config.image_config['model_path'])
            if model_path.exists():
                return joblib.load(model_path)
            else:
                # Initialize new model
                if self.config.image_config['classifier_type'] == 'random_forest':
                    return RandomForestClassifier(
                        **self.config.image_config['random_forest_config']
                    )
                else:
                    return LogisticRegression(
                        **self.config.image_config['logistic_regression_config']
                    )
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            return None
            
    def load_image(self, image_path: Union[str, Path]) -> Optional[np.ndarray]:
        """Load and preprocess image for analysis."""
        try:
            # Read image
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            return image
            
        except Exception as e:
            logging.error(f"Error loading image: {e}")
            return None
            
    def predict(self, image: Union[str, Path, np.ndarray]) -> Dict[str, Union[str, float]]:
        """Predict whether an image is AI-generated.
        
        Args:
            image: Image path or numpy array
            
        Returns:
            Dictionary containing prediction and confidence
        """
        try:
            # Load image if path provided
            if isinstance(image, (str, Path)):
                image = self.load_image(image)
                
            if image is None:
                return {
                    'prediction': 'error',
                    'confidence': 0.0,
                    'error': 'Failed to load image'
                }
                
            # Extract features
            features = self.noise_analyzer.extract_features(image)
            if features is None:
                return {
                    'prediction': 'error',
                    'confidence': 0.0,
                    'error': 'Failed to extract features'
                }
                
            # Convert features to array
            feature_array = np.array([[
                features[feat] for feat in sorted(features.keys())
            ]])
            
            # Get prediction and probability
            prediction = self.model.predict(feature_array)[0]
            probabilities = self.model.predict_proba(feature_array)[0]
            confidence = float(max(probabilities))
            
            # Convert to string label
            label = "AI-generated" if prediction == 1 else "real"
            
            result = {
                'prediction': label,
                'confidence': confidence,
                'features': features
            }
            
            # Add warning if confidence is low
            if confidence < self.config.output_config['confidence_threshold']:
                result['warning'] = 'Low confidence prediction'
                
            return result
            
        except Exception as e:
            logging.error(f"Error during prediction: {e}")
            return {
                'prediction': 'error',
                'confidence': 0.0,
                'error': str(e)
            }
            
    def train(self, images: list, labels: list) -> bool:
        """Train the classifier on a dataset of images.
        
        Args:
            images: List of image paths or arrays
            labels: List of labels (1 for AI-generated, 0 for real)
            
        Returns:
            True if training successful, False otherwise
        """
        try:
            features_list = []
            valid_labels = []
            
            # Extract features from all images
            for img, label in zip(images, labels):
                if isinstance(img, (str, Path)):
                    img = self.load_image(img)
                if img is None:
                    continue
                    
                features = self.noise_analyzer.extract_features(img)
                if features is not None:
                    features_list.append([
                        features[feat] for feat in sorted(features.keys())
                    ])
                    valid_labels.append(label)
            
            if not features_list:
                logging.error("No valid features extracted for training")
                return False
                
            # Convert to numpy arrays
            X = np.array(features_list)
            y = np.array(valid_labels)
            
            # Train model
            self.model.fit(X, y)
            
            # Save model
            save_path = Path(self.config.image_config['model_path'])
            save_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self.model, save_path)
            
            logging.info(f"Model trained and saved to {save_path}")
            return True
            
        except Exception as e:
            logging.error(f"Error during training: {e}")
            return False
