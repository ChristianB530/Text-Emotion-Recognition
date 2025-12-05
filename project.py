import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

# Deep Learning imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


class EmotionDataLoader:
    """Loader for emotion recognition datasets"""
    
    def __init__(self):
        self.label_map = {
            0: 'sadness',
            1: 'joy', 
            2: 'love',
            3: 'anger',
            4: 'fear',
            5: 'surprise'
        }
    
    def load_emotion_dataset(self):
        """Load the Emotion Dataset from Kaggle with 6 labels"""
        try:
            train_df = pd.read_csv('training.csv')
            val_df = pd.read_csv('validation.csv')
            test_df = pd.read_csv('test.csv')
            
            print("Emotion Dataset loaded successfully!")
            print(f"Training samples: {len(train_df):,}")
            print(f"Validation samples: {len(val_df):,}")
            print(f"Test samples: {len(test_df):,}")
            
            # Show dataset statistics
            #self._show_dataset_stats(train_df, "Training")
            #self._show_dataset_stats(val_df, "Validation")
            #self._show_dataset_stats(test_df, "Test")
            
            return train_df, val_df, test_df
            
        except FileNotFoundError as e:
            print(f"Error loading dataset: {e}")
            return None, None, None
    

class BaselineModel:
    """
    Baseline Logistic Regression Model
    Simple, interpretable, and provides solid performance baseline
    """
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        self.model = None
        self.label_encoder = LabelEncoder()
    
    def prepare_features(self, texts, fit=True):
        """Convert texts to TF-IDF features"""
        if fit:
            features = self.vectorizer.fit_transform(texts)
        else:
            features = self.vectorizer.transform(texts)
        return features
    
    def train(self, X_train, y_train, X_val, y_val):
        """Train the logistic regression baseline model"""
        print("Training Logistic Regression Baseline...")
        
        self.model = LogisticRegression(
            random_state=42,
            max_iter=1000,
            multi_class='multinomial',
            class_weight='balanced',
            C=1.0,
            n_jobs=-1
        )
        
        self.model.fit(X_train, y_train)
        
        # Validation performance
        y_pred_val = self.model.predict(X_val)
        val_accuracy = accuracy_score(y_val, y_pred_val)
        val_f1 = f1_score(y_val, y_pred_val, average='macro')
        
        print(f"Baseline Model Performance:")
        print(f"  Validation Accuracy: {val_accuracy:.4f}")
        print(f"  Validation Macro F1: {val_f1:.4f}")
        
        return self.model
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        return self.model.predict_proba(X)
    

# Initialize and load data
loader = EmotionDataLoader()
train_df, val_df, test_df = loader.load_emotion_dataset()

# Initialize BaselineModel
baseline = BaselineModel()

y_train = baseline.label_encoder.fit_transform(train_df['label'])
y_val = baseline.label_encoder.transform(val_df['label'])
y_test = baseline.label_encoder.transform(test_df['label'])
X_train = baseline.prepare_features(train_df['text'], fit=True)
X_val   = baseline.prepare_features(val_df['text'], fit=False)
X_test  = baseline.prepare_features(test_df['text'], fit=False)

model = baseline.train(X_train, y_train, X_val, y_val)

# Evaluate on test set
y_pred_test = baseline.predict(X_test)
target_names = [str(label) for label in baseline.label_encoder.inverse_transform(sorted(np.unique(y_test)))]
print("\n=== Baseline Classification Report ===")
print(classification_report(y_test, y_pred_test, target_names=target_names))


# Predict on new data
sample_text = ["I am so sad! Lost my job today."]
sample_features = baseline.prepare_features(sample_text, fit=False)
prediction = baseline.predict(sample_features)
probabilities = baseline.predict_proba(sample_features)
print(probabilities)





############################################################################################################

class TextPreprocessor:
    """Text preprocessing for emotion recognition"""
    
    def __init__(self):
        self.contractions = {
            "don't": "do not", "can't": "cannot", "won't": "will not",
            "it's": "it is", "i'm": "i am", "you're": "you are",
            "they're": "they are", "we're": "we are", "that's": "that is",
            "what's": "what is", "where's": "where is", "how's": "how is",
            "i've": "i have", "you've": "you have", "we've": "we have",
            "they've": "they have", "i'll": "i will", "you'll": "you will"
        }
    
    def clean_text(self, text):
        """Clean and preprocess text data"""
        # Handle missing values
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Expand contractions
        # Replaces contractions with full forms ("can't" → "cannot")
        for cont, expanded in self.contractions.items():
            text = text.replace(cont, expanded)
        
        # Remove special characters but keep basic punctuation
        # For emotion recognition, punctuation like "!" and "?" can carry emotional cues so we keep them
        text = re.sub(r'[^a-zA-Z\s!?.,]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def preprocess_dataset(self, df, text_column='text'):
        """Preprocess entire dataset"""
        print("Preprocessing text data...")
        df_clean = df.copy()   # Create a copy to avoid modifying original
        df_clean[text_column] = df_clean[text_column].apply(self.clean_text)
        print("Text preprocessing completed!")
        return df_clean

####################################################################################

class Vocabulary:
    """
    Builds a vocabulary mapping words to integers.
    Reserved tokens: 
    0: <PAD> (padding for short sentences)
    1: <UNK> (unknown words)
    """
    def __init__(self, freq_threshold=2, max_size=5000):
        self.itos = {0: "<PAD>", 1: "<UNK>"} # debugging
        self.stoi = {"<PAD>": 0, "<UNK>": 1}
        self.freq_threshold = freq_threshold
        self.max_size = max_size

    def build_vocabulary(self, sentence_list):
        frequencies = Counter()
        idx = 2 # Starting index for actual words (1 & 0 are reserved)
        
        for sentence in sentence_list:
            for word in sentence.split():
                frequencies[word] += 1
                
        # Sort by frequency and limit size
        common_words = frequencies.most_common(self.max_size - 2)
        
        # Assign indices to words
        for word, count in common_words:
            if count >= self.freq_threshold: # Only add words above frequency threshold
                self.stoi[word] = idx
                self.itos[idx] = word
                idx+=1
                
    def numericalize(self, text):
        tokenized_text = [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in text.split()
        ]
        return tokenized_text

class EmotionDataset(Dataset):
    """
    PyTorch Dataset class for Emotion Recognition.
    """
    def __init__(self, df, vocab, max_len=50, is_test=False):
        self.df = df
        self.vocab = vocab
        self.max_len = max_len
        self.is_test = is_test
        
        # Assume the dataframe has 'text' and 'label' columns
        self.texts = df['text'].tolist()
        if not self.is_test:
            self.labels = df['label'].tolist()
            # Assume labels are already encoded as integers
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        text = self.texts[index]
        
        # Convert text to integers
        tokenized_indices = self.vocab.numericalize(text)
        
        # Pad or Truncate
        if len(tokenized_indices) < self.max_len:
            # Pad with 0s if too short
            padded = tokenized_indices + [0] * (self.max_len - len(tokenized_indices))
        else:
            # Truncate if too long
            padded = tokenized_indices[:self.max_len]
            
        # Convert to Tensor
        text_tensor = torch.tensor(padded, dtype=torch.long)
        
        if self.is_test:
            return text_tensor
        else:
            label = self.labels[index]
            return text_tensor, torch.tensor(label, dtype=torch.long)

# Clean the text
preprocessor = TextPreprocessor()
train_df_clean = preprocessor.preprocess_dataset(train_df)
val_df_clean = preprocessor.preprocess_dataset(val_df)
test_df_clean = preprocessor.preprocess_dataset(test_df)

# Ensure labels are integers
train_df_clean['label'] = train_df_clean['label'].astype(int)
val_df_clean['label'] = val_df_clean['label'].astype(int)
test_df_clean['label'] = test_df_clean['label'].astype(int)

# Build vocabulary
print("\nBuilding Vocabulary...")
vocab = Vocabulary(max_size=5000)
vocab.build_vocabulary(train_df_clean['text'].tolist())
print(f"Vocabulary size: {len(vocab.stoi)}")

# Create datasets & loaders
train_dataset = EmotionDataset(train_df_clean, vocab, max_len=50)
val_dataset = EmotionDataset(val_df_clean, vocab, max_len=50)
test_dataset = EmotionDataset(test_df_clean, vocab, max_len=50)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

class EmotionLSTM(nn.Module):
    """
    LSTM RNN Architecture.
    Switched to Bidirectional LSTM for better context capture.
    """
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, n_layers, dropout):
        super(EmotionLSTM, self).__init__()
        
        # Embedding layer: Converts word indices to dense vectors
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # LSTM layer: Bidirectional LSTM
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=n_layers, batch_first=True, dropout=dropout, bidirectional=True)
        
        # Fully connected layer at end
        # Since it's bidirectional, multiply hidden_dim by 2
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        # text shape: [batch size, sent len]
        
        # embedded shape: [batch size, sent len, embed dim]
        embedded = self.dropout(self.embedding(text))
        
        # LSTM returns: output, (hidden_state, cell_state)
        output, (hidden, cell) = self.lstm(embedded)
        
        # Concatenate the final forward and backward hidden states
        # last_hidden shape: [batch size, hidden dim * 2]
        last_hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        
        # Pass through the linear layer
        return self.fc(self.dropout(last_hidden))


# Model Training & Evaluation
def train_model(model, iterator, optimizer, criterion, device):
    """
    Training loop for one epoch
    """
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    
    for batch in iterator:
        # Unpack batch and move to device
        text, labels = batch
        text = text.to(device)
        labels = labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(text)
        
        # Calculate Loss
        loss = criterion(predictions, labels)
        
        # Backward pass
        loss.backward()

        # Clip gradients to prevent gradient explosion 
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

        # Update weights
        optimizer.step()
        
        # Track metrics
        epoch_loss += loss.item()
        
        # Calculate accuracy for monitoring
        _, preds = torch.max(predictions, 1)
        correct = (preds == labels).float()
        acc = correct.sum() / len(correct)
        epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate_model(model, iterator, criterion, device):
    """
    Evaluation loop - No gradient updates
    """
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in iterator:
            text, labels = batch
            text = text.to(device)
            labels = labels.to(device)
            
            predictions = model(text)
            loss = criterion(predictions, labels)
            
            epoch_loss += loss.item()
            
            _, preds = torch.max(predictions, 1)
            correct = (preds == labels).float()
            acc = correct.sum() / len(correct)
            epoch_acc += acc.item()
            
        # Store for F1 calculation
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
            
    # Calculate Macro F1 Score
    f1 = f1_score(all_labels, all_preds, average='macro')
            
    return epoch_loss / len(iterator), epoch_acc / len(iterator), f1

# Hyperparameters
INPUT_DIM = len(vocab.stoi)
EMBED_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = 6 # 6 Emotions
N_LAYERS = 1
DROPOUT = 0.5
N_EPOCHS = 15
LEARNING_RATE = 0.001

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Initialize Model
rnn_model = EmotionLSTM(INPUT_DIM, EMBED_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, DROPOUT)
rnn_model = rnn_model.to(device)

# Optimizer & Loss
optimizer = optim.Adam(rnn_model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()
criterion = criterion.to(device)

# Training Loop
print(f"\nStarting training for {N_EPOCHS} epochs...")
best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):
    train_loss, train_acc = train_model(rnn_model, train_loader, optimizer, criterion, device)
    valid_loss, valid_acc, valid_f1 = evaluate_model(rnn_model, val_loader, criterion, device)
    
    # Save the best model state
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(rnn_model.state_dict(), 'rnn_model.pt')
    
    print(f'Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Val Loss: {valid_loss:.3f} | Val Acc: {valid_acc*100:.2f}% | Val F1: {valid_f1:.3f}')

# Final Evaluation
print("\nLoading best model for testing...")
rnn_model.load_state_dict(torch.load('rnn_model.pt'))

# Get test predictions
rnn_model.eval()
test_preds = []
test_labels = []

with torch.no_grad():
    for text, labels in test_loader:
        text = text.to(device)
        predictions = rnn_model(text)
        _, preds = torch.max(predictions, 1)
        test_preds.extend(preds.cpu().numpy())
        test_labels.extend(labels.cpu().numpy())

emotion_names = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

# Print final report
print("\n=== RNN Classification Report ===")
print(classification_report(test_labels, test_preds, target_names=emotion_names))