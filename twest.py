import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import string
import torch
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
import seaborn as sns
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
import os
import logging
import torch.nn.functional as F
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoConfig,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup
)
from torch import amp  # Add this import for new GradScaler
from torch.optim import AdamW
class ModelConfig:
    def __init__(self):
        self.use_traditional_models = False  # Set to False to skip traditional models
        self.use_bert = True  # Set to True to use BERT
        self.use_ensemble = False  # Set to False to skip ensemble
        self.train_bert = False  # Set to False to skip BERT training
        self.use_saved_bert = True  # Set to True to load saved BERT model
        
# Create global config
CONFIG = ModelConfig()
CONFIG.use_traditional_models = True  # Skip traditional models
CONFIG.use_bert = True  # Use only BERT
CONFIG.use_ensemble = True  # Skip ensemble
CONFIG.use_saved_bert = False  # Load saved BERT model
CONFIG.train_bert = True  # train BERT
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Refactor repetitive code into reusable functions
def log_device_info():
    """Log CUDA device information."""
    logging.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logging.info(f"CUDA device count: {torch.cuda.device_count()}")
        logging.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")

log_device_info()

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Update device setting
if torch.cuda.is_available():
    torch.cuda.empty_cache()  # Clear GPU memory
    device = torch.device("cuda:0")  # Explicitly select first GPU
    logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    logging.info("Using CPU")

# Text preprocessing function for traditional models
def preprocess_text(text):
    if isinstance(text, str):
        # Convert to lowercase
        text = text.lower()
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        # Remove user mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    else:
        return ""

# Load the dataset 
try:
    df = pd.read_csv(r"kagglehub\datasets\muhammadimran112233\liar-twitter-dataset\versions\1\Liar_Dataset.csv")
except FileNotFoundError as e:
    logging.error(f"Dataset file not found: {e}")
    raise
except Exception as e:
    logging.error(f"Error loading dataset: {e}")
    raise

# Check available columns
logging.info("\nAvailable columns:")
logging.info(df.columns.tolist())

# Identify text and label columns
text_column = 'statement'  # Based on the CSV file
label_column = 'label'

# Verify columns exist
if text_column not in df.columns or label_column not in df.columns:
    raise ValueError(f"Required columns not found. Available columns: {df.columns.tolist()}")

# Clean text data
df[text_column] = df[text_column].fillna("")

# Map labels - assuming binary classification (real/fake)
label_mapping = {
    'TRUE': 0,  # Real news
    'FALSE': 1, # Fake news
    'half-true': 0,
    'barely-true': 1,
    'pants-fire': 1,
    'mostly-true': 0 
}

df[label_column] = df[label_column].map(label_mapping)

# Drop rows with missing labels
df = df.dropna(subset=[label_column])

# Verify we have data after preprocessing
logging.info(f"\nNumber of samples after preprocessing: {len(df)}")
logging.info(f"Label distribution:\n{df[label_column].value_counts()}")

# Preprocess text
df['processed_text'] = df[text_column].apply(preprocess_text)

# Add text length feature
df['text_length'] = df['processed_text'].apply(len)

# Split features and target
X = df[['processed_text', 'text_length', text_column]]
 # Include both processed and original text
y = df[label_column]

if len(X) == 0 or len(y) == 0:
    raise ValueError("No data available for training after preprocessing")

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

logging.info(f"\nTraining set size: {len(X_train)}")
logging.info(f"Test set size: {len(X_test)}")

# Create a pipeline for text feature
text_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=5000,
        min_df=5,
        max_df=0.7,
        ngram_range=(1, 2),  # Use both unigrams and bigrams
        stop_words='english'
    ))
])

# Transform text data for traditional models
X_train_text_features = text_pipeline.fit_transform(X_train['processed_text'])
X_test_text_features = text_pipeline.transform(X_test['processed_text'])

# Convert text_length to numpy array and reshape
X_train_length = X_train['text_length'].values.reshape(-1, 1)
X_test_length = X_test['text_length'].values.reshape(-1, 1)

# Combine text features and length feature for traditional models
X_train_combined = np.hstack((X_train_text_features.toarray(), X_train_length))
X_test_combined = np.hstack((X_test_text_features.toarray(), X_test_length))

# Initialize accuracies
nb_accuracy = 0
lr_accuracy = 0
rf_accuracy = 0
ensemble_accuracy = 0

if CONFIG.use_traditional_models:
    logging.info("\nTraining traditional models...")
    
    # Naive Bayes
    nb_model = MultinomialNB()
    nb_model.fit(X_train_combined, y_train)
    nb_pred = nb_model.predict(X_test_combined)
    nb_accuracy = accuracy_score(y_test, nb_pred)
    logging.info(f"Naive Bayes Accuracy: {nb_accuracy:.4f}")
    
    # Logistic Regression
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train_combined, y_train)
    lr_pred = lr_model.predict(X_test_combined)
    lr_accuracy = accuracy_score(y_test, lr_pred)
    logging.info(f"Logistic Regression Accuracy: {lr_accuracy:.4f}")
    
    # Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_combined, y_train)
    rf_pred = rf_model.predict(X_test_combined)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    logging.info(f"Random Forest Accuracy: {rf_accuracy:.4f}")
else:
    nb_model = None
    lr_model = None
    rf_model = None
    logging.info("Skipping traditional models training")

# Ensemble model
if CONFIG.use_ensemble and CONFIG.use_traditional_models:
    ensemble_model = VotingClassifier(
        estimators=[
            ('nb', MultinomialNB()),
            ('lr', LogisticRegression(max_iter=1000, random_state=42)),
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
        ],
        voting='soft'
    )
    ensemble_model.fit(X_train_combined, y_train)
    ensemble_pred = ensemble_model.predict(X_test_combined)
    ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
    logging.info(f"Ensemble Model Accuracy: {ensemble_accuracy:.4f}")
else:
    ensemble_model = None
    logging.info("Skipping ensemble model training")

# BERT Implementation
logging.info("\nImplementing BERT model...")

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Function to prepare data for BERT
def prepare_data_for_bert(texts, labels=None, max_length=128):
    """Prepare data for BERT model."""
    input_ids, attention_masks = [], []
    for text in tqdm(texts, desc="Processing texts"):
        encoded_dict = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',  # Updated argument
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    if labels is not None:
        labels = torch.tensor(labels.values)
        return input_ids, attention_masks, labels
    return input_ids, attention_masks

# Prepare BERT training data
logging.info("Preparing training data for BERT...")
train_input_ids, train_attention_masks, train_labels = prepare_data_for_bert(
    X_train[text_column].values, 
    y_train
)

# Prepare BERT test data
logging.info("Preparing test data for BERT...")
test_input_ids, test_attention_masks, test_labels = prepare_data_for_bert(
    X_test[text_column].values, 
    y_test
)

# Create DataLoader for training and validation
batch_size = 16

train_data = TensorDataset(train_input_ids, train_attention_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

test_data = TensorDataset(test_input_ids, test_attention_masks, test_labels)
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

# Enhanced BERT Configuration
BERT_CONFIG = {
    'model_name': 'bert-base-uncased',  # Changed to BERT base
    'max_length': 512,
    'batch_size': 16,  # Increased since BERT base is smaller
    'accumulation_steps': 4,
    'learning_rate': 2e-5,
    'epochs': 6,
    'warmup_ratio': 0.1,
    'weight_decay': 0.01,
    'dropout': 0.3,
    'scheduler': 'cosine',
    'save_dir': './bert_fake_news_model/',
    'mixed_precision': True,
    'max_grad_norm': 1.0,
    'checkpoint_interval': 1000
}


# Initialize mixed precision training
scaler = amp.GradScaler('cuda', enabled=BERT_CONFIG['mixed_precision'])

# Improved BERT training function
def train_bert_model(model, train_dataloader, validation_dataloader=None, epochs=BERT_CONFIG['epochs']):
    """Enhanced BERT training with gradient accumulation and mixed precision"""
    logging.info(f"Training BERT model ({BERT_CONFIG['model_name']})...")
    
    optimizer = AdamW(
        model.parameters(),
        lr=BERT_CONFIG['learning_rate'],
        weight_decay=BERT_CONFIG['weight_decay']
    )
    
    total_steps = len(train_dataloader) * epochs // BERT_CONFIG['accumulation_steps']
    warmup_steps = int(total_steps * BERT_CONFIG['warmup_ratio'])
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    best_val_loss = float('inf')
    best_model_state = None
    early_stopping_counter = 0
    step_counter = 0
    
    for epoch in range(epochs):
        logging.info(f"\nEpoch {epoch+1}/{epochs}")
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        
        progress_bar = tqdm(train_dataloader, desc="Training")
        for batch_idx, batch in enumerate(progress_bar):
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_attention_mask, b_labels = batch
            
            # Mixed precision forward pass
            with torch.cuda.amp.autocast(enabled=BERT_CONFIG['mixed_precision']):
                outputs = model(
                    b_input_ids,
                    attention_mask=b_attention_mask,
                    labels=b_labels
                )
                loss = outputs.loss / BERT_CONFIG['accumulation_steps']
            
            # Mixed precision backward pass
            scaler.scale(loss).backward()
            total_loss += loss.item() * BERT_CONFIG['accumulation_steps']
            
            # Gradient accumulation
            if (batch_idx + 1) % BERT_CONFIG['accumulation_steps'] == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), BERT_CONFIG['max_grad_norm'])
                
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                
                step_counter += 1
                
                # Checkpoint saving
                if step_counter % BERT_CONFIG['checkpoint_interval'] == 0:
                    checkpoint = {
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'scaler_state_dict': scaler.state_dict(),
                        'step': step_counter,
                        'epoch': epoch
                    }
                    torch.save(checkpoint, f"{BERT_CONFIG['save_dir']}/checkpoint_{step_counter}.pt")
            
            # Update progress bar
            progress_bar.set_description(
                f"Loss: {loss.item()*BERT_CONFIG['accumulation_steps']:.4f} LR: {scheduler.get_last_lr()[0]:.2e}"
            )
        
        # ...rest of the training loop...
        
    return model

# Update model initialization for better memory efficiency
def initialize_bert_model():
    """Initialize BERT model with optimized settings"""
    try:
        # Set memory efficient attention
        config = AutoConfig.from_pretrained(
            BERT_CONFIG['model_name'],
            num_labels=2,
            hidden_dropout_prob=BERT_CONFIG['dropout'],
            attention_probs_dropout_prob=BERT_CONFIG['dropout'],
            use_cache=False  # Disable KV caching for training
        )
        
        model = AutoModelForSequenceClassification.from_pretrained(
            BERT_CONFIG['model_name'],
            config=config
        )
        
        if torch.cuda.is_available():
            model = model.to(device)
            # Enable gradient checkpointing
            model.gradient_checkpointing_enable()
        
        return model
    except Exception as e:
        logging.error(f"Error initializing BERT model: {e}")
        raise

# Update model initialization
bert_model = initialize_bert_model()

# Add after model initialization
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    # Optional: set memory allocation
    torch.cuda.set_per_process_memory_fraction(0.8)  # Use 80% of GPU memory

# Update tokenizer
tokenizer = AutoTokenizer.from_pretrained(BERT_CONFIG['model_name'])

# Improved BERT training function
def train_bert_model(model, train_dataloader, validation_dataloader=None, epochs=BERT_CONFIG['epochs']):
    """Enhanced BERT training with validation and early stopping"""
    logging.info(f"Training BERT model ({BERT_CONFIG['model_name']})...")
    
    # Initialize optimizer with weight decay
    optimizer = AdamW(
        model.parameters(),
        lr=BERT_CONFIG['learning_rate'],
        weight_decay=BERT_CONFIG['weight_decay']
    )
    
    # Calculate total steps
    total_steps = len(train_dataloader) * epochs
    warmup_steps = int(total_steps * BERT_CONFIG['warmup_ratio'])
    
    # Choose scheduler
    if BERT_CONFIG['scheduler'] == 'cosine':
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
    else:
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
    
    # Training loop with early stopping
    best_val_loss = float('inf')
    best_model_state = None
    early_stopping_counter = 0
    patience = 3
    
    for epoch in range(epochs):
        logging.info(f"\nEpoch {epoch+1}/{epochs}")
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_dataloader, desc="Training")
        for batch in progress_bar:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_attention_mask, b_labels = batch
            
            model.zero_grad()
            
            outputs = model(
                b_input_ids,
                attention_mask=b_attention_mask,
                labels=b_labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            # Backward pass with gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            
            # Update progress bar
            progress_bar.set_description(
                f"Loss: {loss.item():.4f} LR: {scheduler.get_last_lr()[0]:.2e}"
            )
        
        avg_train_loss = total_loss / len(train_dataloader)
        logging.info(f"Average training loss: {avg_train_loss:.4f}")
        
        # Validation phase
        if validation_dataloader:
            val_loss = evaluate_bert_model(model, validation_dataloader)['val_loss']
            logging.info(f"Validation loss: {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
            
            if early_stopping_counter >= patience:
                logging.info("Early stopping triggered")
                break
    
    # Load best model if available
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        
    return model

# Check if we should train BERT (can be memory intensive)
bert_epochs = 6  # Reduced for demonstration, use 3-4 for better results
use_saved_model = True  # Set to True if you want to load a saved model
train_bert = False  # Set to False to skip BERT training if needed
def load_saved_model(model_path='./bert_fake_news_model/'):
    """Load a previously saved BERT model and tokenizer"""
    import os
    try:
        logging.info(f"Loading saved model from {model_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path {model_path} not found")
            
        # Load tokenizer and model
        tokenizer = BertTokenizer.from_pretrained(model_path)
        model = BertForSequenceClassification.from_pretrained(model_path)
        
        # Move model to GPU if available
        model.to(device)
        logging.info(f"Model loaded successfully and moved to {device}")
        return model, tokenizer
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        return None, None

# Evaluate BERT model
def evaluate_bert_model(model, dataloader):
    """Evaluate BERT model and return metrics including validation loss"""
    model.eval()
    predictions = []
    true_labels = []
    total_loss = 0
    
    # Evaluate data for one epoch
    logging.info("Evaluating BERT model...")
    for batch in tqdm(dataloader, desc="Evaluating"):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_attention_mask, b_labels = batch
        
        with torch.no_grad():
            # Get both loss and predictions
            outputs = model(
                b_input_ids,
                attention_mask=b_attention_mask,
                labels=b_labels  # Include labels to get loss
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            logits = outputs.logits
            
            # Move predictions to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            
            predictions.append(logits)
            true_labels.append(label_ids)
    
    # Calculate average validation loss
    val_loss = total_loss / len(dataloader)
    
    # Combine all predictions and calculate metrics
    predictions = np.concatenate(predictions, axis=0)
    true_labels = np.concatenate(true_labels, axis=0)
    preds_flat = np.argmax(predictions, axis=1).flatten()
    accuracy = accuracy_score(true_labels, preds_flat)
    
    logging.info(f"BERT Accuracy: {accuracy:.4f}")
    logging.info(f"Validation Loss: {val_loss:.4f}")
    
    return {
        'accuracy': accuracy,
        'predictions': preds_flat,
        'true_labels': true_labels,
        'logits': predictions,
        'val_loss': val_loss  # Make sure val_loss is included in return dict
    }

# Update the model loading and evaluation section:
# Initialize dictionary to store accuracies
all_accuracies = {}

if use_saved_model:
    logging.info("Loading saved BERT model...")
    bert_model, tokenizer = load_saved_model()
    if bert_model is not None:
        logging.info("Evaluating saved BERT model...")
        bert_results = evaluate_bert_model(bert_model, test_dataloader)
        bert_accuracy = bert_results['accuracy']
        bert_predictions = bert_results['predictions']
        logging.info(f"Saved BERT Model Accuracy: {bert_accuracy:.4f}")
        
        # Add BERT to accuracy comparison
        all_accuracies["BERT"] = bert_accuracy
    else:
        logging.info("Failed to load saved model, falling back to traditional models")
        use_saved_model = False
elif train_bert:
    # Original training code...
    bert_model = train_bert_model(bert_model, train_dataloader, epochs=bert_epochs)
    bert_results = evaluate_bert_model(bert_model, test_dataloader)
    bert_accuracy = bert_results['accuracy']
    all_accuracies["BERT"] = bert_accuracy

# Evaluate BERT model
def evaluate_bert_model(model, dataloader):
    """Evaluate BERT model and return metrics including validation loss"""
    model.eval()
    predictions = []
    true_labels = []
    total_loss = 0
    
    # Evaluate data for one epoch
    logging.info("Evaluating BERT model...")
    for batch in tqdm(dataloader, desc="Evaluating"):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_attention_mask, b_labels = batch
        
        with torch.no_grad():
            # Get both loss and predictions
            outputs = model(
                b_input_ids,
                attention_mask=b_attention_mask,
                labels=b_labels  # Include labels to get loss
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            logits = outputs.logits
            
            # Move predictions to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            
            predictions.append(logits)
            true_labels.append(label_ids)
    
    # Calculate average validation loss
    val_loss = total_loss / len(dataloader)
    
    # Combine all predictions and calculate metrics
    predictions = np.concatenate(predictions, axis=0)
    true_labels = np.concatenate(true_labels, axis=0)
    preds_flat = np.argmax(predictions, axis=1).flatten()
    accuracy = accuracy_score(true_labels, preds_flat)
    
    logging.info(f"BERT Accuracy: {accuracy:.4f}")
    logging.info(f"Validation Loss: {val_loss:.4f}")
    
    return {
        'accuracy': accuracy,
        'predictions': preds_flat,
        'true_labels': true_labels,
        'logits': predictions,
        'val_loss': val_loss  # Make sure val_loss is included in return dict
    }

if train_bert:
    bert_results = evaluate_bert_model(bert_model, test_dataloader)
    bert_accuracy = bert_results['accuracy']
    bert_predictions = bert_results['predictions']
    bert_report = classification_report(bert_results['true_labels'], bert_predictions)
    logging.info("\nBERT Classification Report:")
    logging.info(bert_report)
else:
    bert_accuracy = 0
    logging.info("Skipping BERT evaluation as training was disabled")

# Find the best model among all (including BERT if trained)
all_accuracies = {
    "Naive Bayes": nb_accuracy,
    "Logistic Regression": lr_accuracy,
    "Random Forest": rf_accuracy,
    "Ensemble Model": ensemble_accuracy
}

if train_bert:
    all_accuracies["BERT"] = bert_accuracy

best_model_name = max(all_accuracies, key=all_accuracies.get)
best_accuracy = all_accuracies[best_model_name]

logging.info(f"\nBest model: {best_model_name} with accuracy: {best_accuracy:.4f}")

# Create prediction functions for BERT
def predict_with_bert(text: str, model=bert_model, tokenizer=tokenizer) -> dict:
    """Enhanced BERT prediction with confidence calibration"""
    # Prepare data and explicitly move to GPU
    encoded_dict = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=BERT_CONFIG['max_length'],
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    # Move input tensors to GPU
    input_ids = encoded_dict['input_ids'].to(device)
    attention_mask = encoded_dict['attention_mask'].to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        logits = outputs.logits
        probabilities = F.softmax(logits, dim=1)[0]
        prediction = torch.argmax(probabilities)
        
        # Temperature scaling for better calibration
        temperature = 1.5
        calibrated_probs = F.softmax(logits / temperature, dim=1)[0]
        
        confidence = calibrated_probs[prediction].item()
        
        return {
            "prediction": "Fake News" if prediction.item() == 1 else "Real News",
            "confidence": confidence,
            "probability_fake": calibrated_probs[1].item(),
            "probability_real": calibrated_probs[0].item(),
            "model_name": BERT_CONFIG['model_name']
        }

def split_text_into_chunks(text, tokenizer, max_length=512):
    """Split text into overlapping chunks that BERT can process."""
    # Tokenize the full text
    tokens = tokenizer.tokenize(text)
    total_length = len(tokens)
    
    # If text is short enough, return it as a single chunk
    if total_length <= max_length - 2:  # -2 for [CLS] and [SEP] tokens
        return [text]
    
    # Calculate chunk size with overlap
    chunk_size = max_length - 2  # Effective chunk size
    overlap = chunk_size // 10  # 10% overlap to maintain context
    stride = chunk_size - overlap
    
    # Split into chunks
    chunks = []
    for i in range(0, total_length, stride):
        chunk_tokens = tokens[i:i + chunk_size]
        chunk_text = tokenizer.convert_tokens_to_string(chunk_tokens)
        chunks.append(chunk_text)
    
    return chunks

def predict_with_bert(text: str, model=bert_model, tokenizer=tokenizer) -> dict:
    """Enhanced BERT prediction with long text support and confidence calibration"""
    # Split text into chunks if needed
    chunks = split_text_into_chunks(text, tokenizer, BERT_CONFIG['max_length'])
    chunk_predictions = []
    chunk_confidences = []
    
    model.eval()
    for chunk in chunks:
        # Prepare data and explicitly move to GPU
        encoded_dict = tokenizer.encode_plus(
            chunk,
            add_special_tokens=True,
            max_length=BERT_CONFIG['max_length'],
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        
        # Move input tensors to GPU
        input_ids = encoded_dict['input_ids'].to(device)
        attention_mask = encoded_dict['attention_mask'].to(device)
        
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            logits = outputs.logits
            probabilities = F.softmax(logits, dim=1)[0]
            prediction = torch.argmax(probabilities)
            
            # Temperature scaling for better calibration
            temperature = 1.5
            calibrated_probs = F.softmax(logits / temperature, dim=1)[0]
            
            chunk_predictions.append(prediction.item())
            chunk_confidences.append(calibrated_probs)
    
    # Aggregate results from all chunks
    fake_count = sum(1 for p in chunk_predictions if p == 1)
    real_count = len(chunk_predictions) - fake_count
    
    # Calculate weighted vote based on confidences
    total_fake_conf = sum(conf[1].item() for conf in chunk_confidences)
    total_real_conf = sum(conf[0].item() for conf in chunk_confidences)
    
    # Make final prediction
    if total_fake_conf > total_real_conf:
        final_prediction = "Fake News"
        confidence = total_fake_conf / (total_fake_conf + total_real_conf)
        probability_fake = total_fake_conf / len(chunks)
        probability_real = total_real_conf / len(chunks)
    else:
        final_prediction = "Real News"
        confidence = total_real_conf / (total_fake_conf + total_real_conf)
        probability_fake = total_fake_conf / len(chunks)
        probability_real = total_real_conf / len(chunks)
    
    # Include chunk analysis in results
    chunk_analysis = {
        "total_chunks": len(chunks),
        "fake_chunks": fake_count,
        "real_chunks": real_count,
        "fake_confidence_avg": total_fake_conf / len(chunks),
        "real_confidence_avg": total_real_conf / len(chunks)
    }
    
    return {
        "prediction": final_prediction,
        "confidence": confidence,
        "probability_fake": probability_fake,
        "probability_real": probability_real,
        "model_name": BERT_CONFIG['model_name'],
        "chunk_analysis": chunk_analysis
    }

# Function to predict with traditional model
def predict_with_traditional(text: str, model, pipeline) -> dict:
    # Preprocess the text
    processed = preprocess_text(text)
    text_length = len(processed)
    
    # Transform text using the pipeline
    text_features = pipeline.transform([processed])
    
    # Combine with length feature
    length_feature = np.array([text_length]).reshape(1, -1)
    combined_features = np.hstack((text_features.toarray(), length_feature))
    
    # Get prediction and probability
    prediction = model.predict(combined_features)[0]
    probabilities = model.predict_proba(combined_features)[0]
    
    # Determine result and confidence
    label = "Fake News" if prediction == 1 else "Real News"
    confidence = probabilities[prediction]
    
    return {
        "prediction": label,
        "confidence": confidence,
        "probability_fake": probabilities[1],
        "probability_real": probabilities[0]
    }

# Create a universal prediction function
def predict_fake_news(text, use_bert=True):
    if use_bert and train_bert and best_model_name == "BERT":
        return predict_with_bert(text, bert_model, tokenizer)
    else:
        # Select traditional model
        if best_model_name == "Naive Bayes":
            model = nb_model
        elif best_model_name == "Logistic Regression":
            model = lr_model
        elif best_model_name == "Random Forest":
            model = rf_model
        else:  # Ensemble model
            model = ensemble_model
            
        return predict_with_traditional(text, model, text_pipeline)

# Update ComprehensiveFakeNewsDetector initialization
class ComprehensiveFakeNewsDetector:
    def __init__(self, config=None):
        self.config = config or CONFIG
        
        # Initialize models based on configuration
        if self.config.use_traditional_models:
            self.nb_model = nb_model
            self.lr_model = lr_model
            self.rf_model = rf_model
            self.text_pipeline = text_pipeline
        else:
            self.nb_model = None
            self.lr_model = None
            self.rf_model = None
            self.text_pipeline = None
            
        if self.config.use_ensemble:
            self.ensemble_model = ensemble_model
        else:
            self.ensemble_model = None
            
        if self.config.use_bert:
            self.bert_model = bert_model
            self.tokenizer = tokenizer
            self.use_bert = True
        else:
            self.bert_model = None
            self.tokenizer = None
            self.use_bert = False
        
        # Determine best model based on available models
        self.best_model_name = self._determine_best_model()
        
    def _determine_best_model(self):
        accuracies = {}
        if self.config.use_traditional_models:
            if self.nb_model: accuracies["Naive Bayes"] = nb_accuracy
            if self.lr_model: accuracies["Logistic Regression"] = lr_accuracy
            if self.rf_model: accuracies["Random Forest"] = rf_accuracy
        if self.config.use_ensemble:
            if self.ensemble_model: accuracies["Ensemble"] = ensemble_accuracy
        if self.config.use_bert:
            if self.bert_model: accuracies["BERT"] = bert_accuracy
            
        return max(accuracies.items(), key=lambda x: x[1])[0] if accuracies else "BERT"
    
    def predict(self, text, model_name=None):
        """Predict using specified model or best model if not specified"""
        if model_name is None:
            model_name = self.best_model_name
            
        if model_name == "BERT" and self.bert_model is not None:
            result = predict_with_bert(text, self.bert_model, self.tokenizer)
            logging.info(f"Using BERT model for prediction: {result['prediction']}")
            return result
        else:
            # Use traditional models
            if model_name == "Naive Bayes":
                model = self.nb_model
            elif model_name == "Logistic Regression":
                model = self.lr_model
            elif model_name == "Random Forest":
                model = self.rf_model
            else:  # Ensemble model
                model = self.ensemble_model
            
            result = predict_with_traditional(text, model, self.text_pipeline)
            logging.info(f"Using {model_name} for prediction: {result['prediction']}")
            return result
    
    def predict_with_all_models(self, text):
        """Predict using all available models"""
        results = {}
        
        # Only add traditional models if they exist
        if self.ensemble_model is not None:
            results["Ensemble Model"] = predict_with_traditional(text, self.ensemble_model, self.text_pipeline)
            
            # Optional traditional models if they exist
            if self.nb_model is not None:
                results["Naive Bayes"] = predict_with_traditional(text, self.nb_model, self.text_pipeline)
            if self.lr_model is not None:
                results["Logistic Regression"] = predict_with_traditional(text, self.lr_model, self.text_pipeline)
            if self.rf_model is not None:
                results["Random Forest"] = predict_with_traditional(text, self.rf_model, self.text_pipeline)
        
        # Add BERT model if available
        if self.use_bert and self.bert_model is not None:
            results["BERT"] = predict_with_bert(text, self.bert_model, self.tokenizer)
        
        if not results:
            raise ValueError("No models available for prediction")
        
        return results
    
    def majority_vote(self, text):
        """Get prediction based on majority vote from all models"""
        all_predictions = self.predict_with_all_models(text)
        votes = {"Real News": 0, "Fake News": 0}
        
        for model_name, result in all_predictions.items():
            votes[result["prediction"]] += 1
        
        # Determine winner
        if votes["Fake News"] > votes["Real News"]:
            return "Fake News", votes["Fake News"] / sum(votes.values())
        elif votes["Real News"] > votes["Fake News"]:
            return "Real News", votes["Real News"] / sum(votes.values())
        else:
            # Tie - use confidence as tiebreaker
            fake_confidence = sum([r["probability_fake"] for _, r in all_predictions.items()])
            real_confidence = sum([r["probability_real"] for _, r in all_predictions.items()])
            
            if fake_confidence > real_confidence:
                return "Fake News", fake_confidence / (fake_confidence + real_confidence)
            else:
                return "Real News", real_confidence / (fake_confidence + real_confidence)
    
    def weighted_ensemble(self, text, weights=None):
        """Get prediction based on weighted ensemble of all models"""
        all_predictions = self.predict_with_all_models(text)
        
        # Default weights with fallback values if accuracies are not available
        if weights is None:
            weights = {}
            # Assign default weights based on empirical performance
            default_weights = {
                "Ensemble Model": 0.25,
                "BERT": 0.3,
                "Naive Bayes": 0.15,
                "Logistic Regression": 0.15,
                "Random Forest": 0.15
            }
            
            # Try to use actual accuracies if available, otherwise use defaults
            for model_name in all_predictions.keys():
                if model_name == "Ensemble Model" and self.ensemble_model is not None:
                    weights[model_name] = ensemble_accuracy if ensemble_accuracy > 0 else default_weights[model_name]
                elif model_name == "BERT" and self.bert_model is not None:
                    weights[model_name] = bert_accuracy if bert_accuracy > 0 else default_weights[model_name]
                elif model_name == "Naive Bayes" and self.nb_model is not None:
                    weights[model_name] = nb_accuracy if nb_accuracy > 0 else default_weights[model_name]
                elif model_name == "Logistic Regression" and self.lr_model is not None:
                    weights[model_name] = lr_accuracy if lr_accuracy > 0 else default_weights[model_name]
                elif model_name == "Random Forest" and self.rf_model is not None:
                    weights[model_name] = rf_accuracy if rf_accuracy > 0 else default_weights[model_name]
        
        # Only use weights for available models
        weights = {k: v for k, v in weights.items() if k in all_predictions}
        
        if not weights:
            # If no weights are available, use equal weights
            weights = {k: 1.0/len(all_predictions) for k in all_predictions.keys()}
        else:
            # Normalize weights
            total_weight = sum(weights.values())
            weights = {k: v/total_weight for k, v in weights.items()}
        
        # Calculate weighted probabilities
        fake_prob = sum([p["probability_fake"] * weights[model] 
                         for model, p in all_predictions.items() 
                         if model in weights])
        real_prob = sum([p["probability_real"] * weights[model] 
                         for model, p in all_predictions.items() 
                         if model in weights])
        
        # Determine prediction
        if fake_prob > real_prob:
            return "Fake News", fake_prob / (fake_prob + real_prob)
        else:
            return "Real News", real_prob / (fake_prob + real_prob)

# Create detector instance
detector = ComprehensiveFakeNewsDetector()  # Pass the global config

# Test the comprehensive prediction system with different samples
test_texts = [
    "Breaking: Scientists discover a new planet with intelligent life!",
    "The stock market closed higher today after positive economic reports.",
    "Local authorities confirm multiple sightings of flying unicorns.",
    "COVID-19 vaccines have been tested extensively before approval."
]

logging.info("\nTesting with the best model:")
for text in test_texts:
    result = detector.predict(text)
    logging.info(f"\nText: {text}")
    logging.info(f"Prediction ({detector.best_model_name}): {result['prediction']} with {result['confidence']*100:.2f}% confidence")

if len(detector.predict_with_all_models(test_texts[0])) > 1:  # If we have multiple models
    logging.info("\nTesting with majority vote ensemble:")
    for text in test_texts:
        prediction, confidence = detector.majority_vote(text)
        logging.info(f"\nText: {text}")
        logging.info(f"Majority Vote Prediction: {prediction} with {confidence*100:.2f}% agreement")
    
    logging.info("\nTesting with weighted ensemble:")
    for text in test_texts:
        prediction, confidence = detector.weighted_ensemble(text)
        logging.info(f"\nText: {text}")
        logging.info(f"Weighted Ensemble Prediction: {prediction} with {confidence*100:.2f}% confidence")

# Class to capture the entire pipeline
class FakeNewsDetectionSystem:
    def __init__(self, data_path=None, load_pretrained=True, pretrained_path='./bert_fake_news_model/'):
        self.detector = None
        import os
        if load_pretrained and os.path.exists(pretrained_path):
            try:
                # Load BERT model from saved directory
                bert_model = BertForSequenceClassification.from_pretrained(pretrained_path)
                tokenizer = BertTokenizer.from_pretrained(pretrained_path)
                bert_model.to(device)
                
                # Update config and create detector
                config = ModelConfig()
                config.use_bert = True
                config.use_traditional_models = False
                config.use_ensemble = False
                
                # Create detector with loaded model
                self.detector = ComprehensiveFakeNewsDetector(config)
                self.detector.bert_model = bert_model
                self.detector.tokenizer = tokenizer
                
                logging.info("Loaded pre-trained models successfully")
            except Exception as e:
                logging.error(f"Failed to load pre-trained models: {e}")
                self.train_new_system(data_path)
        else:
            self.train_new_system(data_path)
    
    def train_new_system(self, data_path):
        # This would implement the full training pipeline
        # For now, we'll use the already trained detector
        self.detector = detector
        logging.info("Using already trained detector as new system")
    
    def predict(self, text, method="best"):
        """
        Predict if text is fake news using different methods
        
        Parameters:
        text (str): Text to classify
        method (str): One of "best", "majority", "weighted", "all"
        
        Returns:
        dict: Prediction results
        """
        if not self.detector:
            raise ValueError("Detector not initialized. Please train or load a model first.")
        
        if method == "best":
            return self.detector.predict(text)
        elif method == "majority":
            prediction, confidence = self.detector.majority_vote(text)
            return {"prediction": prediction, "confidence": confidence}
        elif method == "weighted":
            prediction, confidence = self.detector.weighted_ensemble(text)
            return {"prediction": prediction, "confidence": confidence}
        elif method == "all":
            return self.detector.predict_with_all_models(text)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def evaluate(self, texts, labels):
        """Evaluate the system on provided texts and labels"""
        predictions = [self.predict(text)["prediction"] == "Fake News" for text in texts]
        labels = [bool(label) for label in labels]  # Convert to boolean
        
        accuracy = sum([p == l for p, l in zip(predictions, labels)]) / len(labels)
        return {
            "accuracy": accuracy,
            "predictions": predictions,
            "true_labels": labels
        }
    
    def save(self, path="./saved_fake_news_system_2/"):
        """Save the entire detection system"""
        import pickle
        import os
        
        if not os.path.exists(path):
            os.makedirs(path)
        
        # Save traditional models
        with open(os.path.join(path, "traditional_models.pkl"), "wb") as f:
            pickle.dump({
                "nb_model": self.detector.nb_model,
                "lr_model": self.detector.lr_model,
                "rf_model": self.detector.rf_model,
                "ensemble_model": self.detector.ensemble_model,
                "text_pipeline": self.detector.text_pipeline
            }, f)
        
        # Save BERT if available
        if hasattr(self.detector, "bert_model") and self.detector.bert_model is not None:
            bert_path = os.path.join(path, "bert_model")
            if not os.path.exists(bert_path):
                os.makedirs(bert_path)
            
            self.detector.bert_model.save_pretrained(bert_path)
            self.detector.tokenizer.save_pretrained(bert_path)
        
        # Save configuration
        with open(os.path.join(path, "config.pkl"), "wb") as f:
            pickle.dump({
                "best_model_name": self.detector.best_model_name,
                "use_bert": hasattr(self.detector, "bert_model") and self.detector.bert_model is not None
            }, f)
        
        logging.info(f"System saved to {path}")
    
    @classmethod
    def load(cls, path="./saved_fake_news_syste_2/"):
        """Load a saved detection system"""
        import pickle
        import os
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path not found: {path}")
        
        system = cls(load_pretrained=True)
        system.detector = ComprehensiveFakeNewsDetector(use_bert=True)
        
        # Load traditional models
        with open(os.path.join(path, "traditional_models.pkl"), "rb") as f:
            models = pickle.load(f)
            system.detector.nb_model = models["nb_model"]
            system.detector.lr_model = models["lr_model"]
            system.detector.rf_model = models["rf_model"]
            system.detector.ensemble_model = models["ensemble_model"]
            system.detector.text_pipeline = models["text_pipeline"]
        
        # Load configuration
        with open(os.path.join(path, "config.pkl"), "rb") as f:
            config = pickle.load(f)
            system.detector.best_model_name = config["best_model_name"]
        
        # Load BERT if available
        bert_path = os.path.join(path, "bert_model")
        if os.path.exists(bert_path) and config.get("use_bert", False):
            system.detector.bert_model = BertForSequenceClassification.from_pretrained(bert_path)
            system.detector.tokenizer = BertTokenizer.from_pretrained(bert_path)
            system.detector.bert_model.to(device)
            system.detector.use_bert = True
        
        logging.info(f"System loaded from {path}")
        return system

# Create the full system
system = FakeNewsDetectionSystem()

# Example of using the full system
logging.info("\nTesting the complete system with different methods:")
text = "Researchers discover that coffee causes immortality, according to a new study."

logging.info("\nBest model prediction:")
logging.info(system.predict(text, method="best"))

if train_bert or len(detector.predict_with_all_models(text)) > 1:
    logging.info("\nMajority vote prediction:")
    logging.info(system.predict(text, method="majority"))
    
    logging.info("\nWeighted ensemble prediction:")
    logging.info(system.predict(text, method="weighted"))
    
    logging.info("\nAll models predictions:")
    all_predictions = system.predict(text, method="all")
    for model_name, prediction in all_predictions.items():
        logging.info(f"{model_name}: {prediction['prediction']} with {prediction['confidence']*100:.2f}% confidence")

# Example of saving the system (comment out if not needed)
# system.save()

# Fix device logging at the end of the file
logging.info(f"Model device: {next(bert_model.parameters()).device}")

# Create a sample input to check device
sample_text = "This is a test input"
encoded_dict = tokenizer.encode_plus(
    sample_text,
    add_special_tokens=True,
    max_length=512,
    padding='max_length',
    return_attention_mask=True,
    return_tensors='pt'
)
logging.info(f"Input tensor device: {encoded_dict['input_ids'].device}")

if CONFIG.use_bert:
    logging.info("\nInitializing BERT...")
    if CONFIG.use_saved_bert:
        try:
            bert_model = BertForSequenceClassification.from_pretrained('./bert_fake_news_model/')
            tokenizer = BertTokenizer.from_pretrained('./bert_fake_news_model/')
            bert_model.to(device)
            logging.info("Loaded saved BERT model")
        except Exception as e:
            logging.error(f"Error loading saved BERT model: {e}")
            CONFIG.train_bert = True  # Fall back to training if loading fails
    
    if CONFIG.train_bert:
        bert_model = initialize_bert_model()
        bert_model = train_bert_model(bert_model, train_dataloader, test_dataloader)
        logging.info("Completed BERT training")
else:
    bert_model = None
    tokenizer = None
    logging.info("Skipping BERT initialization")