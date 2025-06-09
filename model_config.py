class ModelConfig:
    """Configuration settings for fake news detection models.
    
    This includes:
    - Which models to use (BERT, traditional ML, ensemble)  
    - Model-specific parameters and training settings
    - Text preprocessing configuration
    """
 
    def __init__(self):
        # Model selection        self.use_traditional_models = False  # Enable traditional ML models
        self.use_bert = True               # Enable BERT model
        self.use_ensemble = False           # Enable ensemble methods
        self.train_bert = False           # Whether to train BERT from scratch
        self.use_saved_bert = True         # Use the configured save directory
        self.use_traditional_models = False  # Enable traditional ML models
        
        
        self.bert_config = {
            'model_name': 'bert-base-uncased',
            'max_length': 128,              # Reduced sequence length to save memory
            'batch_size': 8,                # Smaller batch size for 3050 Ti
            'accumulation_steps': 8,        # Increased to maintain effective batch size of 64
            'learning_rate': 2e-5,
            'epochs': 1,
            'gradient_checkpointing': True,  # Memory optimization            'fp16': True,                   # Enable mixed precision training
            'save_dir': './saved_model/bert',  # Directory to save the model
            'dropout': 0.1,                 # Regularization
            'warmup_ratio': 0.1,           # Warmup steps ratio
            'weight_decay': 0.01,          # L2 regularization
            'max_grad_norm': 1.0,          # Gradient clipping
            
            
            # GPU optimization settings
            'device': 'cuda',              # Use CUDA for training
            'num_workers': 4,              # DataLoader workers for faster data loading
            'pin_memory': True,            # Pin memory for faster data transfer to GPU
            'gradient_checkpointing': True, # Trade compute for memory
            'optimizer_type': 'adamw',      # Use AdamW optimizer
            'adam_epsilon': 1e-8,          # AdamW epsilon parameter
            'max_grad_norm': 1.0,          # Gradient clipping norm
            'scheduler_type': 'linear',     # Learning rate scheduler type
            'save_steps': 500,             # Save checkpoint every N steps
            'eval_steps': 500,             # Evaluate every N steps
            'logging_steps': 100,          # Log metrics every N steps
            'save_total_limit': 2,         # Keep only last N checkpoints
        }
        
        # Traditional models configuration
        self.traditional_config = {
            'naive_bayes': {
                'alpha': 1.0,
            },
            'logistic_regression': {
                'max_iter': 1000,
                'C': 1.0
            },
            'random_forest': {
                'n_estimators': 100,
                'max_depth': None
            }
        }
        
        # Text preprocessing configuration
        self.preprocessing_config = {
            'remove_stopwords': True,
            'lemmatize': True,
            'min_df': 5,
            'max_df': 0.95,
            'max_features': 10000
        }
        
        # Training configuration
        self.training_config = {
            'train_test_split': 0.2,
            'random_state': 42,
            'early_stopping': True,
            'patience': 3,
            'eval_steps': 100
        }
        
        # Initialize preprocessing pipeline
        self.text_pipeline = None
        self._setup_preprocessing()
        
    def _setup_preprocessing(self):
        """Setup the text preprocessing pipeline."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.pipeline import Pipeline
            from fake_news_detection.utils.preprocessing import TextPreprocessor
            
            self.text_pipeline = Pipeline([
                ('preprocessor', TextPreprocessor(
                    remove_stopwords=self.preprocessing_config['remove_stopwords'],
                    lemmatize=self.preprocessing_config['lemmatize']
                )),
                ('vectorizer', TfidfVectorizer(
                    min_df=self.preprocessing_config['min_df'],
                    max_df=self.preprocessing_config['max_df'],
                    max_features=self.preprocessing_config['max_features']
                ))
            ])
        except Exception as e:
            import logging
            logging.error(f"Failed to setup preprocessing pipeline: {e}")
            self.text_pipeline = None
