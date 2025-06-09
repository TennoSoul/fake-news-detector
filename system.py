"""High level interface for the fake news detection system."""

import os
import pickle
import logging
from typing import Optional, Dict, List, Any, Union
import torch
from transformers import BertForSequenceClassification, BertTokenizer, BertConfig
import pandas as pd

from fake_news_detection.models.detector import ComprehensiveFakeNewsDetector
from fake_news_detection.models.traditional_models import TraditionalModel
from fake_news_detection.config.model_config import ModelConfig

class FakeNewsDetectionSystem:
    """High level interface for fake news detection.
    
    This class:
    1. Manages model loading/saving and configuration
    2. Provides convenient prediction methods
    3. Handles model training and evaluation
    4. Takes care of model persistence
    5. Manages ensemble methods and model combination
    """
    def __init__(self, config: Optional[ModelConfig] = None,
                 data_path: Optional[str] = None,
                 load_pretrained: bool = True,
                 pretrained_path: str = './bert_fake_news_model/'): 
        """Initialize the system with optimized GPU settings."""
        self.config = config if config is not None else ModelConfig()
        self.data_path = data_path
        
        # Optimize CUDA settings if available
        if torch.cuda.is_available():
            try:
                # Get available memory on each GPU
                free_memory = []
                for i in range(torch.cuda.device_count()):
                    torch.cuda.set_device(i)
                    torch.cuda.empty_cache()
                    free_memory.append(torch.cuda.get_device_properties(i).total_memory - 
                                     torch.cuda.memory_allocated(i))
                
                # Select GPU with most free memory
                device_id = free_memory.index(max(free_memory))
                torch.cuda.set_device(device_id)
                self.device = torch.device(f"cuda:{device_id}")
                
                # Configure optimized CUDA settings
                if self.config.bert_config.get('fp16', True):
                    # Enable TF32 for better performance on Ampere GPUs
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                
                # Set optimal CUDNN settings
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
                
                # Set memory allocation settings
                memory_fraction = self.config.bert_config.get('cuda_memory_fraction', 0.85)
                torch.cuda.set_per_process_memory_fraction(memory_fraction)
                
                # Log GPU info
                logging.info(f"Using GPU device {device_id}: {torch.cuda.get_device_name(device_id)}")
                logging.info(f"CUDA Memory allocated: {torch.cuda.memory_allocated(device_id) / 1024**2:.2f}MB")
                logging.info(f"CUDA Memory reserved: {torch.cuda.memory_reserved(device_id) / 1024**2:.2f}MB")
                
            except Exception as e:
                logging.error(f"Error initializing CUDA: {e}. Falling back to CPU.")
                self.device = torch.device('cpu')
        else:
            self.device = torch.device('cpu')
            logging.info("No GPU available, using CPU")

        # Initialize detector with optimized device settings
        self.detector = ComprehensiveFakeNewsDetector(self.config, device=self.device)
        
        if load_pretrained and os.path.exists(pretrained_path):
            try:
                # Load BERT with optimized settings if enabled
                if self.config.use_bert:
                    logging.info(f"Loading BERT model on {self.device}")
                    bert_config = BertConfig.from_pretrained(
                        pretrained_path,
                        num_labels=2,
                        hidden_dropout_prob=self.config.bert_config['dropout'],
                        attention_probs_dropout_prob=self.config.bert_config['dropout']
                    )
                    bert_model = BertForSequenceClassification.from_pretrained(
                        pretrained_path,
                        config=bert_config,
                        num_labels=2
                    )
                    tokenizer = BertTokenizer.from_pretrained(pretrained_path)
                      # Initialize detector with device
                    self.detector = ComprehensiveFakeNewsDetector(self.config, device=self.device)
                    
                    # Set up multi-GPU if available
                    if torch.cuda.device_count() > 1:
                        logging.info(f"Using {torch.cuda.device_count()} GPUs")
                        bert_model = torch.nn.DataParallel(bert_model)
                    
                    # Update BERT model and tokenizer
                    self.detector.bert_model.model = bert_model
                    self.detector.bert_model.tokenizer = tokenizer
                    
                    # Clear CUDA cache after model loading
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    logging.info("Loaded pre-trained BERT model with GPU optimizations")
                    
                # Load traditional models if enabled
                if self.config.use_traditional_models:
                    traditional_path = os.path.join(pretrained_path, "traditional_models.pkl")
                    if os.path.exists(traditional_path):
                        with open(traditional_path, "rb") as f:
                            models_dict = pickle.load(f)
                        
                        if not self.detector:
                            self.detector = ComprehensiveFakeNewsDetector(self.config)
                            
                        self.detector.load_traditional_models(models_dict)
                        logging.info("Loaded pre-trained traditional models")
                        
            except Exception as e:
                logging.error(f"Error loading pretrained models: {str(e)}")
                self.detector = ComprehensiveFakeNewsDetector(self.config)
        else:
            self.detector = ComprehensiveFakeNewsDetector(self.config)
            
    def train(self, train_data: pd.DataFrame, validation_data: Optional[pd.DataFrame] = None):
        """Train the detection system.
        
        Args:
            train_data: Training data
            validation_data: Optional validation data
        """
        if self.device.type == 'cuda':
            try:
                # Clear GPU cache before training
                torch.cuda.empty_cache()
                
                # Monitor initial GPU state
                init_memory = torch.cuda.memory_allocated() / 1024**2
                logging.info(f"Initial GPU memory usage: {init_memory:.2f}MB")
                
                # Set up gradient scaler for mixed precision training
                scaler = torch.cuda.amp.GradScaler(enabled=self.config.bert_config['fp16'])
                
                # Optimize GPU memory usage
                train_kwargs = {
                    'pin_memory': self.config.bert_config.get('pin_memory', True),
                    'num_workers': self.config.bert_config.get('num_workers', 2),
                    'persistent_workers': True if self.config.bert_config.get('num_workers', 2) > 0 else False,
                    'prefetch_factor': 2
                }
                
                # Enable gradient checkpointing if configured
                grad_checkpointing = self.config.bert_config.get('gradient_checkpointing', False)
                
            except Exception as e:
                logging.error(f"Error configuring CUDA training: {e}")
                # Fall back to CPU training with minimal settings
                scaler = None
                train_kwargs = {}
                grad_checkpointing = False
        else:
            scaler = None
            train_kwargs = {}
            grad_checkpointing = False
            
        if not self.detector:
            self.detector = ComprehensiveFakeNewsDetector(config=self.config, device=self.device)
            
        try:
            self.detector.train(
                train_data, 
                validation_data=validation_data,
                device=self.device,
                scaler=scaler,
                gradient_checkpointing=grad_checkpointing,
                **train_kwargs
            )
            
            if self.device.type == 'cuda':
                # Log memory stats after training
                final_memory = torch.cuda.memory_allocated() / 1024**2
                logging.info(f"Final GPU memory usage: {final_memory:.2f}MB")
                logging.info(f"Memory change during training: {final_memory - init_memory:.2f}MB")
                
                # Clear cache after training
                torch.cuda.empty_cache()
                
        except Exception as e:
            logging.error(f"Error during training: {e}")
            raise
            
    def train_new_system(self, data_path: str):
        """Train a new detection system from scratch.
        
        Args:
            data_path: Path to training data
        """
        try:
            # Load and prepare data
            logging.info("Loading training data...")
            texts, labels = self._load_training_data(data_path)
            
            from fake_news_detection.utils.training import prepare_data, train_model, evaluate_model
            
            data_splits = prepare_data(
                texts, 
                labels,
                test_size=self.config.training_config['train_test_split'],
                val_size=0.2,
                random_state=self.config.training_config['random_state']
            )
            
            # Initialize detector
            self.detector = ComprehensiveFakeNewsDetector(self.config)
            
            # Train BERT if enabled
            if self.config.use_bert:
                logging.info("Training BERT model...")
                train_model(
                    self.detector.bert_model,
                    train_data=data_splits['train'],
                    val_data=data_splits['val']
                )
                
            # Train traditional models if enabled
            if self.config.use_traditional_models:
                for model_name in self.detector.traditional_models:
                    logging.info(f"Training {model_name} model...")
                    model = self.detector.traditional_models[model_name]
                    train_model(
                        model,
                        train_data=data_splits['train']  
                    )
                    
            # Final evaluation
            logging.info("Evaluating on test set...")
            test_results = evaluate_model(
                self.detector,
                test_data=data_splits['test'],
                save_path=os.path.join(data_path, "evaluation")
            )
            
            logging.info(f"Training completed. Test accuracy: {test_results['metrics']['accuracy']:.4f}")
            
        except Exception as e:
            logging.error(f"Failed to train new system: {e}")
            raise
            
    def _load_training_data(self, data_path: str) -> tuple[List[str], List[int]]:
        """Load training data from files.
        
        Args:
            data_path: Path to training data directory
            
        Returns:
            Tuple of (texts, labels)
        """
        try:
            import pandas as pd
            
            # Try different file formats
            if os.path.exists(os.path.join(data_path, "training_data.csv")):
                df = pd.read_csv(os.path.join(data_path, "training_data.csv"))
            elif os.path.exists(os.path.join(data_path, "training_data.json")):
                df = pd.read_json(os.path.join(data_path, "training_data.json"))
            else:
                raise FileNotFoundError("No training data file found")
                
            # Validate data format
            required_cols = {"text", "label"}
            if not required_cols.issubset(df.columns):
                raise ValueError(f"Data must contain columns: {required_cols}")
                
            texts = df["text"].tolist()
            labels = df["label"].astype(int).tolist()
            
            logging.info(f"Loaded {len(texts)} training examples")
            return texts, labels
            
        except Exception as e:
            logging.error(f"Failed to load training data: {e}")
            raise
    def predict(self, text: str, method: str = "best") -> Dict[str, Any]:
            
        """Predict whether text is fake news using different methods.
        
        Args:
            text: Text to classify
            method: One of "best", "majority", "weighted", "all"
                - best: Use the best performing model
                - majority: Use majority voting from all models
                - weighted: Use weighted ensemble of all models
                - all: Return predictions from all models
            
        Returns:
            Prediction dictionary with:
                - prediction: "Fake News" or "Real News"
                - confidence: Prediction confidence score
                - probability_fake: Probability of being fake
                - probability_real: Probability of being real
                - (optional) model_predictions: Individual model predictions
        """
        if not self.detector:
            raise RuntimeError("Detector not initialized. Please load or train models first.")
            
        try:
            if method == "best":
                result = self.detector.predict(text)
            elif method == "majority":
                pred, conf = self.detector.majority_vote(text)
                result = {
                    "prediction": pred,
                    "confidence": conf,
                    "method": "majority_vote"
                }
            elif method == "weighted":
                pred, conf = self.detector.weighted_ensemble(text)
                result = {
                    "prediction": pred,
                    "confidence": conf,
                    "method": "weighted_ensemble"
                }
            elif method == "all":
                all_preds = self.detector.predict_with_all_models(text)
                # Get best model prediction as main result
                best_pred = all_preds[self.detector.best_model_name]
                result = {
                    **best_pred,
                    "model_predictions": all_preds
                }
            else:
                raise ValueError(f"Unknown prediction method: {method}")
                
            return result
            
        except Exception as e:
            logging.error(f"Prediction failed: {e}")
            return {
                "prediction": "Error",
                "confidence": 0.0,
                "error": str(e)
            }
            
    def evaluate(self, texts: List[str], labels: List[int],
                method: str = "best",
                save_path: Optional[str] = None) -> Dict[str, Any]:
        """Evaluate system performance comprehensively.
        
        Args:
            texts: Test texts
            labels: Binary labels (1 for fake, 0 for real)
            method: Prediction method to evaluate
            save_path: Optional path to save evaluation results
            
        Returns:
            Dictionary with evaluation metrics including:
                - accuracy: Overall accuracy
                - precision: Precision for fake news detection
                - recall: Recall for fake news detection
                - f1_score: F1 score
                - confusion_matrix: Confusion matrix
                - classification_report: Detailed classification metrics
                - predictions: Model predictions
                - true_labels: True labels
        """
        if not self.detector:
            raise RuntimeError("Detector not initialized. Please load or train models first.")
            
        try:
            from fake_news_detection.utils.training import evaluate_model
            
            # Get predictions using specified method
            predictions = []
            for text in texts:
                result = self.predict(text, method=method)
                predictions.append(
                    1 if result["prediction"] == "Fake News" else 0
                )
                
            # Calculate metrics
            results = evaluate_model(
                self.detector,
                test_data=(texts, labels),
                save_path=save_path
            )
            
            # Add method-specific info
            results['evaluation_method'] = method
            results['model_config'] = {
                'use_bert': self.config.use_bert,
                'use_traditional_models': self.config.use_traditional_models,
                'use_ensemble': self.config.use_ensemble
            }
            
            return results
            
        except Exception as e:
            logging.error(f"Evaluation failed: {e}")
            raise    
    
        
    def save(self, path: str = "./saved_fake_news_system/"):
            """Save the entire detection system.
            
            Args:
                path: Directory path where to save the model and its components
            """
            import pickle
            import os
            
            # Convert relative path to absolute if needed
            if not os.path.isabs(path):
                path = os.path.abspath(path)
                
            # Create base directory if it doesn't exist
            if not os.path.exists(path):
                os.makedirs(path)
                
            try:
                    # Save config
                with open(os.path.join(path, "config.pkl"), "wb") as f:
                    pickle.dump({
                        "best_model_name": self.detector.best_model_name if hasattr(self.detector, "best_model_name") else None,
                        "use_bert": hasattr(self.detector, "bert_model") and self.detector.bert_model is not None
                    }, f)
                
                # Save traditional models if they exist
                with open(os.path.join(path, "traditional_models.pkl"), "wb") as f:
                    pickle.dump({
                        "nb_model": self.detector.nb_model if hasattr(self.detector, "nb_model") else None,
                        "lr_model": self.detector.lr_model if hasattr(self.detector, "lr_model") else None,
                        "rf_model": self.detector.rf_model if hasattr(self.detector, "rf_model") else None,
                        "ensemble_model": self.detector.ensemble_model if hasattr(self.detector, "ensemble_model") else None,
                        "text_pipeline": self.detector.text_pipeline if hasattr(self.detector, "text_pipeline") else None
                    }, f)
                
                # Save BERT model if available
                if hasattr(self.detector, "bert_model") and self.detector.bert_model is not None:
                    # The bert_model.save() method handles path resolution internally
                    self.detector.bert_model.save()
                
                logging.info(f"System saved to {path}")
                
            except Exception as e:
                logging.error(f"Error saving system to {path}: {e}")
                raise
    
    @classmethod
    def load(cls, path: str = "./my_saved_model/") -> 'FakeNewsDetectionSystem':
        """Load a saved detection system.
        
        Args:
            path: Path to saved system
            
        Returns:
            Loaded FakeNewsDetectionSystem instance
            
        Raises:
            FileNotFoundError: If path doesn't exist
            RuntimeError: If loading fails
        """
        path = os.path.abspath(path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Saved system not found at {path}")
            
        try:
            # Load configuration
            with open(os.path.join(path, "config.pkl"), "rb") as f:
                config = pickle.load(f)
        except Exception as e:
            logging.error(f"Failed to load configuration: {e}")
                    
            # Create system instance
            system = cls(config=config, load_pretrained=True)
            system.detector = ComprehensiveFakeNewsDetector(config)
            
            # Load metadata if available
            try:
                with open(os.path.join(path, "metadata.pkl"), "rb") as f:
                    metadata = pickle.load(f)
                    system.detector.best_model_name = metadata["best_model_name"]
                    if "model_accuracies" in metadata:
                        system.detector.model_accuracies = metadata["model_accuracies"]
            except FileNotFoundError:
                logging.warning("No metadata file found")
                
            # Load traditional models if enabled
            if config.use_traditional_models:
                traditional_path = os.path.join(path, "traditional_models.pkl")
                if os.path.exists(traditional_path):
                    with open(traditional_path, "rb") as f:
                        models_dict = pickle.load(f)
                        
                    for name, model_data in models_dict.items():
                        model_config = model_data['config']
                        model = TraditionalModel(
                            model_type=model_config['model_type'],
                            text_pipeline=config.text_pipeline
                        )
                        model.model = model_data['model']
                        system.detector.traditional_models[name] = model
                        
                    logging.info("Loaded traditional models")
                    
            # Load BERT if enabled
            if config.use_bert:
                bert_path = os.path.join(path, "bert_model")
                if os.path.exists(bert_path):
                    bert_model = BertForSequenceClassification.from_pretrained(
                        bert_path,
                        num_labels=2
                    )
                    tokenizer = BertTokenizer.from_pretrained(bert_path)
                    bert_model.to(system.device)
                    
                    system.detector.bert_model.model = bert_model
                    system.detector.bert_model.tokenizer = tokenizer
                    
                    logging.info("Loaded BERT model")
                    
        logging.info(f"Successfully loaded system from {path}")
        logging.info(f"System loaded from {path}")
        return system

