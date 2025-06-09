"""Main detector class that combines multiple models for fake news detection."""

import logging
from typing import Optional, Dict, List, Any, Union
import torch
from fake_news_detection.models.bert_model import BERTModel
from fake_news_detection.models.traditional_models import TraditionalModel
from fake_news_detection.config.model_config import ModelConfig

class ComprehensiveFakeNewsDetector:
    """Combines multiple models for fake news detection.
    
    This class can use:
    - BERT model for deep learning predictions
    - Traditional ML models (Naive Bayes, Logistic Regression, Random Forest)
    - Ensemble methods (majority voting, weighted ensemble)
    """
    
    def __init__(self, config: Optional[ModelConfig] = None, device: Optional[torch.device] = None):
        """Initialize the comprehensive detector.
        
        Args:
            config: Configuration for which models to use and their parameters
            device: Device to use for models, inherited from parent system
        """
        self.config = config or ModelConfig()
        self.device = device or torch.device("cpu")
        
        # Initialize models based on config
        self.bert_model: Optional[BERTModel] = None
        self.traditional_models: Dict[str, TraditionalModel] = {}
        
        if self.config.use_bert:
            try:
                self.bert_model = BERTModel(self.config, device=self.device)
                logging.info(f"BERT model initialized successfully on {self.device}")
            except Exception as e:
                logging.error(f"Failed to initialize BERT model: {e}")
                self.bert_model = None

        if self.config.use_traditional_models:
            model_types = [
                "naive_bayes",
                "logistic_regression", 
                "random_forest",
                "ensemble"
            ]
            
            for model_type in model_types:
                try:
                    self.traditional_models[model_type] = TraditionalModel(
                        model_type=model_type,
                        text_pipeline=self.config.text_pipeline
                    )
                    logging.info(f"Initialized {model_type} model")
                except Exception as e:
                    logging.error(f"Failed to initialize {model_type} model: {e}")
                    
        # Determine best model based on validation performance
        self.best_model_name = self._determine_best_model()
        
    def predict(self, text: str, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Predict using specified model or best model.
        
        Args:
            text: Text to classify
            model_name: Name of model to use, or None for best model
            
        Returns:
            Prediction dictionary
        """
        if model_name is None:
            model_name = self.best_model_name
        if model_name == "BERT" and self.bert_model is not None:
            result = self.bert_model.predict(text)
            # Convert numeric label to string prediction
            result['prediction'] = "Fake News" if result['label'] == 1 else "Real News"
            logging.info(f"BERT prediction: {result['prediction']}")
            return result
            
        elif model_name in self.traditional_models:
            result = self.traditional_models[model_name].predict(text)
            # Ensure traditional models follow same format
            if 'prediction' in result and 'label' not in result:
                result['label'] = 1 if result['prediction'] == "Fake News" else 0
            logging.info(f"{model_name} prediction: {result['prediction']}")
            return result
            
        else:
            raise ValueError(f"Model {model_name} not available")

    def predict_with_all_models(self, text: str) -> Dict[str, Dict[str, Any]]:
        """Get predictions from all available models.
        
        Args:
            text: Text to classify
            
        Returns:
            Dictionary mapping model names to their predictions
        """
        results = {}
        
        if self.bert_model is not None:
            results["BERT"] = self.bert_model.predict(text)
            
        for model_name, model in self.traditional_models.items():
            results[model_name] = model.predict(text)
            
        return results

    def majority_vote(self, text: str) -> tuple[str, float]:
        """Get prediction based on majority vote from all models.
        
        Args:
            text: Text to classify
            
        Returns:
            Tuple of (prediction, confidence)
        """
        all_predictions = self.predict_with_all_models(text)
        votes = {"Real News": 0, "Fake News": 0}
        
        # Count votes
        for result in all_predictions.values():
            votes[result["prediction"]] += 1
            
        # Get winner
        total_votes = sum(votes.values())
        if votes["Fake News"] > votes["Real News"]:
            return "Fake News", votes["Fake News"] / total_votes
        elif votes["Real News"] > votes["Fake News"]:
            return "Real News", votes["Real News"] / total_votes
        else:
            # Tiebreaker using confidence
            fake_conf = sum(r["probability_fake"] for r in all_predictions.values())
            real_conf = sum(r["probability_real"] for r in all_predictions.values())
            
            if fake_conf > real_conf:
                return "Fake News", fake_conf / (fake_conf + real_conf)
            else:
                return "Real News", real_conf / (fake_conf + real_conf)

    def weighted_ensemble(self, text: str, weights: Optional[Dict[str, float]] = None) -> tuple[str, float]:
        """Get prediction based on weighted ensemble of all models.
        
        Args:
            text: Text to classify
            weights: Optional dict mapping model names to weights
            
        Returns:
            Tuple of (prediction, confidence)
        """
        all_predictions = self.predict_with_all_models(text)
        
        if weights is None:
            # Equal weights
            weights = {k: 1.0/len(all_predictions) for k in all_predictions.keys()}
        else:
            # Normalize weights
            total = sum(weights.values())
            weights = {k: v/total for k, v in weights.items()}
            
        # Calculate weighted probabilities
        fake_prob = sum(
            p["probability_fake"] * weights[model]
            for model, p in all_predictions.items()
            if model in weights
        )
        
        real_prob = sum(
            p["probability_real"] * weights[model]
            for model, p in all_predictions.items()
            if model in weights
        )
        
        if fake_prob > real_prob:
            return "Fake News", fake_prob / (fake_prob + real_prob)
        else:
            return "Real News", real_prob / (fake_prob + real_prob)
            
    def _determine_best_model(self) -> str:
        """Determine best model based on validation performance.
        
        Returns:
            Name of best performing model
        """
        # For now use BERT if available, otherwise ensemble
        if self.bert_model is not None:
            return "BERT"
        elif "ensemble" in self.traditional_models:
            return "ensemble" 
        else:
            # Return first available model
            available = list(self.traditional_models.keys())
            if available:
                return available[0]
            else:
                raise RuntimeError("No models available")
