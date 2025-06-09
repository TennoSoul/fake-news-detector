"""Implementation of traditional machine learning models for fake news detection."""

import logging
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from fake_news_detection.models.base_model import BaseModel

class TraditionalModel(BaseModel):
    """Wrapper for sklearn-based traditional ML models."""
    
    def __init__(self, model_type="ensemble", text_pipeline=None):
        """Initialize traditional model.
        
        Args:
            model_type: One of "naive_bayes", "logistic_regression", "random_forest", "ensemble"
            text_pipeline: Sklearn pipeline for text preprocessing
        """
        self.model_type = model_type
        self.text_pipeline = text_pipeline
        self.model = None
        
        if model_type == "naive_bayes":
            self.model = MultinomialNB()
        elif model_type == "logistic_regression":
            self.model = LogisticRegression(max_iter=1000)
        elif model_type == "random_forest":
            self.model = RandomForestClassifier(n_estimators=100)
        elif model_type == "ensemble":
            # Voting ensemble of all models
            self.nb_model = MultinomialNB()
            self.lr_model = LogisticRegression(max_iter=1000) 
            self.rf_model = RandomForestClassifier(n_estimators=100)
            self.model = self.nb_model # Main model for predictions
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def predict(self, text: str) -> dict:
        """Predict whether text is fake news."""
        if not self.check_model_health():
            raise RuntimeError("Model not properly initialized")
            
        # Preprocess text
        X = self.text_pipeline.transform([text])
        
        # Get probabilities
        probs = self.model.predict_proba(X)[0]
        pred_class = self.model.predict(X)[0]
        
        # For ensemble, average probabilities
        if self.model_type == "ensemble":
            probs_nb = self.nb_model.predict_proba(X)[0]
            probs_lr = self.lr_model.predict_proba(X)[0]
            probs_rf = self.rf_model.predict_proba(X)[0]
            
            # Average probabilities
            probs = (probs_nb + probs_lr + probs_rf) / 3
            pred_class = 1 if probs[1] > 0.5 else 0
            
        return {
            "prediction": "Fake News" if pred_class == 1 else "Real News",
            "confidence": float(max(probs)),
            "probability_fake": float(probs[1]),
            "probability_real": float(probs[0])
        }

    def batch_predict(self, texts: list) -> list:
        """Predict for a batch of texts."""
        return [self.predict(text) for text in texts]

    def evaluate(self, texts: list, labels: list) -> dict:
        """Evaluate model on texts and labels."""
        predictions = self.batch_predict(texts)
        pred_labels = [1 if p["prediction"] == "Fake News" else 0 for p in predictions]
        
        accuracy = sum([p == l for p, l in zip(pred_labels, labels)]) / len(labels)
        
        return {
            "accuracy": accuracy,
            "predictions": pred_labels,
            "true_labels": labels
        }

    def check_model_health(self) -> bool:
        """Check if model is loaded and working properly."""
        return (
            self.model is not None and 
            self.text_pipeline is not None
        )
