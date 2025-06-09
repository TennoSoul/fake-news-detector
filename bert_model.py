import os
import torch
import logging
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
from fake_news_detection.models.base_model import BaseModel
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from typing import Optional, Dict, List, Union, Any
from tqdm.auto import tqdm
import sys

# Configure logging to output to terminal
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Initialize CUDA if available
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.init()
    logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    logging.info("CUDA not available, using CPU")

class NewsDataset(torch.utils.data.Dataset):
    """Dataset class for news data."""
    def __init__(self, texts, labels, tokenizer, max_length):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )
        # Fix tensor copy warning
        self.labels = torch.tensor(labels, dtype=torch.long)
        
    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item
            
    def __len__(self):
        return len(self.labels)

class BERTModel(BaseModel):
    def __init__(self, config, device: Optional[torch.device] = None):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = device
        self._initialize_model()

    def get_save_dir(self) -> str:
        """Get the absolute path to save/load the model
        
        Ensures consistent path handling across the codebase.
        """
        save_dir = self.config.bert_config['save_dir']
        if not os.path.isabs(save_dir):
            # Convert relative path to absolute
            save_dir = os.path.abspath(save_dir)
        return save_dir

    def save(self) -> None:
        """Save the BERT model and tokenizer to disk using proper path handling"""
        if self.model is None or self.tokenizer is None:
            logging.error("Cannot save model - model or tokenizer is not initialized")
            return

        save_dir = self.get_save_dir()
        try:
            # Create directory if it doesn't exist
            os.makedirs(save_dir, exist_ok=True)
            
            # Save model and tokenizer
            self.model.save_pretrained(save_dir)
            self.tokenizer.save_pretrained(save_dir)
            logging.info(f"Model and tokenizer saved to {save_dir}")
        except Exception as e:
            logging.error(f"Error saving model to {save_dir}: {e}")
            raise

    def _initialize_model(self):
        """Initialize BERT model with optimized settings"""
        try:
            # Initialize tokenizer first
            self.tokenizer = self._load_tokenizer()
            
            # Resolve save directory path
            save_dir = self.get_save_dir()
            
            # Configure model settings
            model_config = BertConfig.from_pretrained(
                save_dir if self.config.use_saved_bert else self.config.bert_config['model_name'],
                num_labels=2,
                hidden_dropout_prob=self.config.bert_config['dropout'],
                attention_probs_dropout_prob=self.config.bert_config['dropout']
            )

            # Load model with optimized settings
            if self.config.use_saved_bert:
                if not os.path.exists(save_dir):
                    raise ValueError(f"Saved model directory {save_dir} does not exist")
                self.model = BertForSequenceClassification.from_pretrained(
                    save_dir,
                    config=model_config,
                    local_files_only=True
                )
            else:
                self.model = BertForSequenceClassification.from_pretrained(
                    self.config.bert_config['model_name'],
                    config=model_config
                )
            
            # Move model to device if specified
            if self.device is not None:
                self.model.to(self.device)
                logging.info(f"BERT model moved to {self.device}")
            
            # Enable gradient checkpointing if configured
            if self.config.bert_config.get('gradient_checkpointing', False):
                self.model.gradient_checkpointing_enable()
                logging.info("Gradient checkpointing enabled")
            
        except Exception as e:
            logging.error(f"Error initializing BERT model: {e}")
            raise

    def to(self, device: torch.device) -> 'BERTModel':
        """Move model to specified device with optimized settings"""
        self.device = device
        if self.model is not None:
            self.model.to(device)
            
            # Enable gradient checkpointing if configured
            if self.config.bert_config.get('gradient_checkpointing', False):
                self.model.gradient_checkpointing_enable()
                
        return self

    def _load_tokenizer(self):
        """Load and configure tokenizer"""
        try:
            save_dir = self.get_save_dir()
            if self.config.use_saved_bert and os.path.exists(save_dir):
                tokenizer = BertTokenizer.from_pretrained(
                    save_dir,
                    local_files_only=True
                )
            else:
                tokenizer = BertTokenizer.from_pretrained(
                    self.config.bert_config['model_name']
                )
            return tokenizer
        except Exception as e:
            logging.error(f"Error loading BERT tokenizer: {e}")
            raise

    @torch.no_grad()
    def predict(self, text: str, device: Optional[torch.device] = None) -> dict:
        """Optimized prediction for single text"""
        if device is not None:
            self.to(device)
            
        self.model.eval()
        
        # Tokenize with padding and truncation
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=self.config.bert_config['max_length'],
            return_tensors="pt"
        )
        
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Use mixed precision if configured and on CUDA
        if self.device.type == 'cuda' and self.config.bert_config.get('fp16', False):
            with autocast():
                outputs = self.model(**inputs)
        else:
            outputs = self.model(**inputs)
            
        logits = outputs.logits
        probs = F.softmax(logits, dim=1)
        
        prediction = {
            'label': int(torch.argmax(probs).item()),
            'probability': float(torch.max(probs).item())
        }
        
        return prediction

    @torch.no_grad()    
    def batch_predict(self, texts: List[str], device: Optional[torch.device] = None, batch_size: int = 32) -> List[Dict[str, Any]]:
        """Optimized batch prediction"""
        if device is not None:
            self.to(device)
            
        self.model.eval()
        all_predictions = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=self.config.bert_config['max_length'],
                return_tensors="pt"
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Use mixed precision if configured and on CUDA
            if self.device.type == 'cuda' and self.config.bert_config.get('fp16', False):
                with autocast():
                    outputs = self.model(**inputs)
            else:
                outputs = self.model(**inputs)
                
            logits = outputs.logits
            probs = F.softmax(logits, dim=1)
              # Convert predictions to list of dicts with consistent labels
            batch_preds = []
            for pred_idx, pred_prob in zip(torch.argmax(probs, dim=1), torch.max(probs, dim=1)[0]):
                label_idx = int(pred_idx.item())
                prob = float(pred_prob.item())
                
                # Convert to string label based on prediction
                if label_idx == 0:
                    label = "TRUE"  # Real news
                else:
                    label = "FALSE"  # Fake news
                    
                batch_preds.append({
                    'label': label,
                    'probability': prob,
                    'confidence': f"{prob * 100:.1f}%"
                })
            
            all_predictions.extend(batch_preds)
            
            # Free GPU memory
            del inputs, outputs, logits, probs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return all_predictions

    def train(self, train_texts: list, train_labels: list,
              val_texts: list = None, val_labels: list = None,
              epochs: int = 1) -> Dict[str, List[float]]:
        """Train the model with optimized settings
        
        Args:
            train_texts: List of training text samples
            train_labels: List of training labels
            val_texts: Optional list of validation text samples
            val_labels: Optional list of validation labels 
            epochs: Number of training epochs
            
        Returns:
            Dict containing training history (losses and metrics)
        """
        # Use config values if parameters not provided
        epochs = epochs if epochs is not None else self.config.bert_config.get('epochs', 1)
        batch_size = self.config.bert_config.get('batch_size', 8)
        
        # Prepare datasets
        train_dataset = NewsDataset(
            train_texts,
            train_labels,
            self.tokenizer,
            self.config.bert_config['max_length']
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        if val_texts and val_labels:
            val_dataset = NewsDataset(
                val_texts,
                val_labels,
                self.tokenizer,
                self.config.bert_config['max_length']
            )
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                pin_memory=True if self.device.type == 'cuda' else False
            )
        
        # Initialize optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.bert_config.get('learning_rate', 2e-5),
            weight_decay=self.config.bert_config.get('weight_decay', 0.01)
        )
        
        # Initialize scheduler
        total_steps = len(train_loader) * epochs
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.config.bert_config.get('learning_rate', 2e-5),
            total_steps=total_steps
        )
        
        # Initialize mixed precision training
        use_amp = self.device.type == 'cuda' and self.config.bert_config.get('fp16', False)
        scaler = torch.amp.GradScaler() if use_amp else None
        
        history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        # Training loop
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            correct = 0
            total = 0
            pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}', 
            unit='batch', leave=True,
            ncols=100,
            position=0,
            postfix={'loss': '?', 'acc': '?'},
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]')
            
            for batch in pbar:
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Clear gradients
                optimizer.zero_grad()
                
                # Forward pass with mixed precision
                if use_amp:
                    with torch.amp.autocast(device_type=self.device.type, dtype=torch.float16):
                        outputs = self.model(**batch)
                        loss = outputs.loss
                else:
                    outputs = self.model(**batch)
                    loss = outputs.loss
                
                # Calculate accuracy
                predictions = torch.argmax(outputs.logits, dim=1)
                correct += (predictions == batch['labels']).sum().item()
                total += len(batch['labels'])
                
                # Backward pass with mixed precision
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                
                scheduler.step()
                total_loss += loss.item()
                
                # Update progress bar
                avg_loss = total_loss / (pbar.n + 1)
                avg_acc = correct / total
                pbar.set_postfix(loss=f'{avg_loss:.3f}', acc=f'{avg_acc:.3f}')
            
            # Calculate training metrics
            avg_train_loss = total_loss / len(train_loader)
            train_accuracy = correct / total
            history['train_loss'].append(avg_train_loss)
            history['train_accuracy'].append(train_accuracy)
            
            # Validation step
            if val_texts and val_labels:
                val_loss, val_accuracy = self._validate(val_loader, use_amp)
                history['val_loss'].append(val_loss)
                history['val_accuracy'].append(val_accuracy)
                
                # Save best model if validation improves
                if len(history['val_accuracy']) == 1 or val_accuracy > max(history['val_accuracy'][:-1]):
                    best_model_dir = os.path.join(self.get_save_dir(), 'best_model')
                    os.makedirs(best_model_dir, exist_ok=True)
                    self.model.save_pretrained(best_model_dir)
                    self.tokenizer.save_pretrained(best_model_dir)
                    logging.info(f"Saved best model to {best_model_dir} (validation accuracy: {val_accuracy:.4f})")
        
        return history

    @torch.no_grad()
    def _validate(self, val_loader, use_amp):
        """Run validation step."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in val_loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            if use_amp:
                with torch.amp.autocast(device_type=self.device.type, dtype=torch.float16):
                    outputs = self.model(**batch)
            else:
                outputs = self.model(**batch)
                
            loss = outputs.loss
            predictions = torch.argmax(outputs.logits, dim=1)
            correct += (predictions == batch['labels']).sum().item()
            total += len(batch['labels'])
            total_loss += loss.item()
        
        return total_loss / len(val_loader), correct / total    
    def save(self, path: Optional[str] = None):
        """Save the model and tokenizer.
        
        Args:
            path: Path to save the model. If None, uses config's save directory.
        """
        if path is None:
            path = self.config.bert_config['save_dir']
        elif not os.path.isabs(path):
            # If path is relative, make it relative to config's save directory
            path = os.path.join(self.config.bert_config['save_dir'], path)
            
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        logging.info(f"Model and tokenizer saved to {path}")

    def evaluate(self, texts: List[str], labels: List[Union[int, float]], device: Optional[torch.device] = None) -> Dict[str, Any]:
        """Evaluate model performance with comprehensive metrics."""
        # Use provided device or fall back to model's device
        eval_device = device if device is not None else self.device
        self.to(eval_device)
          # Get predictions for all texts
        batch_results = self.batch_predict(texts)
        
        # Extract numeric predictions from results
        pred_labels = []
        for result in batch_results:
            # Skip any error results
            if result.get('label') == 'Error':
                logging.warning("Encountered error in prediction during evaluation")
                continue
                
            # Convert string label back to numeric
            if result['label'] in ['TRUE', 'mostly-true', 'half-true']:
                pred_labels.append(0)  # Real news
            else:
                pred_labels.append(1)  # Fake news
        
        # Calculate basic metrics
        correct_predictions = sum(1 for p, l in zip(pred_labels, labels) if p == l)
        accuracy = correct_predictions / len(labels)
        
        # Calculate comprehensive metrics using sklearn
        from sklearn.metrics import precision_recall_fscore_support
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, pred_labels, average='binary', zero_division=0
        )
        
        # Return metrics and predictions
        return {
            'metrics': {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1)
            },
            'predictions': pred_labels
        }
        
    def train_batch(self, batch, scaler=None):
        """Optimized training for one batch"""
        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
          # Forward pass with mixed precision if enabled
        if scaler is not None:
            with torch.amp.autocast(device_type=device_type, dtype=torch.float16):
                outputs = self.model(**batch)
                loss = outputs.loss
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
        else:
            outputs = self.model(**batch)
            loss = outputs.loss
            loss.backward()
            
        return loss.item()
        
    def train_epoch(self, train_loader, optimizer, scheduler=None, scaler=None):
        """Train for one epoch with optimization"""
        self.model.train()
        total_loss = 0
        
        for step, batch in enumerate(train_loader):
            # Clear gradients
            optimizer.zero_grad()
            
            # Train batch with appropriate precision
            loss = self.train_batch(batch, scaler)
            total_loss += loss
            
            # Optimizer step with gradient scaling if enabled
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
                
            # Update learning rate
            if scheduler is not None:
                scheduler.step()
                
        return total_loss / len(train_loader)

    def check_model_health(self) -> bool:
        """Check model health status."""
        try:
            # Check model and tokenizer
            if not self.model or not self.tokenizer:
                logging.error("Model or tokenizer not initialized")
                return False
                
            # Check device placement
            model_device = next(self.model.parameters()).device
            if model_device != self.device:
                logging.error(f"Model on wrong device. Expected {self.device}, got {model_device}")
                return False
            
            # Verify save directory exists and is writable
            save_dir = self.get_save_dir()
            if not os.path.exists(save_dir):
                try:
                    os.makedirs(save_dir)
                except Exception as e:
                    logging.error(f"Cannot create save directory {save_dir}: {e}")
                    return False
            elif not os.access(save_dir, os.W_OK):
                logging.error(f"Save directory {save_dir} is not writable")
                return False
            
            # Test prediction
            test_text = "This is a test article about important world events that needs to be verified."
            logging.info("Running model health check...")
            result = self.predict(test_text)
            
            if "error" in result:
                logging.error(f"Prediction failed: {result['error']}")
                return False
                
            # Check memory usage
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / 1024**2  # MB
                if memory_allocated > 8000:
                    logging.warning(f"High GPU memory usage: {memory_allocated:.2f} MB")
                
            logging.info("Model health check passed")
            return True
            
        except Exception as e:
            logging.error(f"Health check failed: {str(e)}")
            return False
