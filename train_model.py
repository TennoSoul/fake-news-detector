"""Example script for training the fake news detection model."""

import logging
import torch
from fake_news_detection.config.model_config import ModelConfig
from fake_news_detection.system import FakeNewsDetectionSystem
from fake_news_detection.utils.training import prepare_data, train_model, evaluate_model, plot_training_history

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def main():
    setup_logging()    # Initialize configuration with GPU optimizations
    config = ModelConfig()
    config.train_bert = True  # Enable BERT training
    config.use_saved_bert = False  # Don't use pre-trained model
      # GPU settings are already configured in ModelConfig
    # Initialize the system
    system = FakeNewsDetectionSystem(config=config)
    
    # Create output directories
    import os
    os.makedirs('data/training', exist_ok=True)
    os.makedirs('data/evaluation', exist_ok=True)
    
    # Load the LIAR dataset
    import pandas as pd
    
    logging.info("Loading LIAR dataset...")
    dataset_path = os.path.join('kagglehub', 'datasets', 'muhammadimran112233', 
                               'liar-twitter-dataset', 'versions', '1', 'Liar_Dataset.csv')
    
    if not os.path.exists(dataset_path):
        logging.error(f"Dataset not found at {dataset_path}")
        return
        
    try:
        df = pd.read_csv(dataset_path)
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        return
    
    # Convert labels to binary (1 for false/pants-fire/barely-true, 0 for true/mostly-true/half-true)
    label_map = {
        'FALSE': 1, 'pants-fire': 1, 'barely-true': 1,
        'TRUE': 0, 'mostly-true': 0, 'half-true': 0
    }
    
    # Filter out rows with labels not in our mapping
    df = df[df['label'].isin(label_map.keys())]
    
    # Convert labels and get texts
    texts = df['statement'].tolist()
    labels = df['label'].map(label_map).tolist()
    
    logging.info(f"Dataset loaded: {len(texts)} samples")
    logging.info(f"Real news samples: {sum(1 for l in labels if l == 0)}")
    logging.info(f"Fake news samples: {sum(1 for l in labels if l == 1)}")
    
    # Prepare data splits
    data_splits = prepare_data(
        texts=texts,
        labels=labels,
        test_size=0.2,  # 20% for testing
        val_size=0.2,   # 20% of training data for validation
        random_state=42
    )
      # Check GPU availability and set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    if torch.cuda.is_available():
        logging.info(f"GPU: {torch.cuda.get_device_name(0)}")
        # Clear CUDA cache before training
        torch.cuda.empty_cache()
    
    # Move model to appropriate device
    system.detector.bert_model.to(device)
    
    # Train the model
    logging.info("Starting model training...")
    history = train_model(
        model=system.detector.bert_model,
        train_data=(data_splits['train'][0], data_splits['train'][1]),
        val_data=(data_splits['val'][0], data_splits['val'][1]),
        epochs=config.bert_config.get('epochs', 1)
    )
    
    # Plot training history
    plot_training_history(
        history,
        save_path='data/training'
    )
    
    # Evaluate the model
    logging.info("Evaluating model...")
    results = evaluate_model(
        model=system.detector.bert_model,
        test_data=(data_splits['test'][0], data_splits['test'][1]),
        save_path='data/evaluation'
    )
    
    # Print evaluation metrics
    logging.info(f"Test Accuracy: {results['metrics']['accuracy']:.4f}")
    logging.info("\nClassification Report:")
    logging.info(results['metrics']['classification_report'])
      # Save the trained model
    logging.info("Saving model...")
    save_path = config.bert_config['save_dir']
    system.save(save_path)
    logging.info(f"Training complete! Model saved to {save_path}")

if __name__ == "__main__":
    main()
