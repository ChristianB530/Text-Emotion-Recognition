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

# Build Vocabulary
print("\nBuilding Vocabulary...")
vocab = Vocabulary(max_size=5000)
vocab.build_vocabulary(train_df_clean['text'].tolist())
print(f"Vocabulary size: {len(vocab.stoi)}")

# Create Datasets & Loaders
train_dataset = EmotionDataset(train_df_clean, vocab, max_len=50)
val_dataset = EmotionDataset(val_df_clean, vocab, max_len=50)
test_dataset = EmotionDataset(test_df_clean, vocab, max_len=50)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

