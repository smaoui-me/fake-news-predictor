import os
import torch
import numpy as np
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score

from .base_model import BaseModel

class DistilBertModel(BaseModel):
    """
    DistilBERT model for fake news classification.
    """
    
    def __init__(self, max_length=128, batch_size=16, epochs=3, learning_rate=2e-5):
        """
        Initialize the model.
        
        Args:
            max_length: Maximum sequence length for tokenization
            batch_size: Batch size for training
            epochs: Number of training epochs
            learning_rate: Learning rate for optimization
        """
        self.max_length = max_length
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        
        # Initialize tokenizer and model
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = DistilBertForSequenceClassification.from_pretrained(
            'distilbert-base-uncased', num_labels=2
        )
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def _tokenize_data(self, texts):
        """
        Tokenize the text data.
        
        Args:
            texts: List or Series of text data
            
        Returns:
            Dictionary containing input_ids and attention_mask
        """
        return self.tokenizer(
            list(texts),
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
    
    def _create_dataloader(self, texts, labels=None):
        """
        Create a DataLoader for the provided texts and labels.
        
        Args:
            texts: List or Series of text data
            labels: List or Series of labels (0 for real, 1 for fake)
            
        Returns:
            DataLoader object
        """
        tokenized = self._tokenize_data(texts)
        
        if labels is not None:
            # Training data
            dataset = TensorDataset(
                tokenized['input_ids'],
                tokenized['attention_mask'],
                torch.tensor(labels.values if hasattr(labels, 'values') else labels)
            )
        else:
            # Prediction data
            dataset = TensorDataset(
                tokenized['input_ids'],
                tokenized['attention_mask']
            )
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=labels is not None  # Shuffle only for training
        )
    
    def train(self, texts, labels):
        """
        Train the model on the provided texts and labels.
        
        Args:
            texts: List or Series of text data
            labels: List or Series of labels (0 for real, 1 for fake)
        """
        # Create DataLoader
        train_dataloader = self._create_dataloader(texts, labels)
        
        # Prepare optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        total_steps = len(train_dataloader) * self.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            
            for batch in train_dataloader:
                # Unpack the batch and move to device
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device)
                labels = batch[2].to(self.device)
                
                # Forward pass
                self.model.zero_grad()
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
            
            avg_loss = total_loss / len(train_dataloader)
            print(f"Epoch {epoch+1}/{self.epochs} - Average Loss: {avg_loss:.4f}")
        
        return self
    
    def predict(self, texts):
        """
        Make predictions on the provided texts.
        
        Args:
            texts: List or Series of text data
            
        Returns:
            numpy array of predictions (0 for real, 1 for fake)
        """
        # Create DataLoader
        dataloader = self._create_dataloader(texts)
        
        # Prediction loop
        self.model.eval()
        all_predictions = []
        
        with torch.no_grad():
            for batch in dataloader:
                # Unpack the batch and move to device
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                # Get predictions
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=1).cpu().numpy()
                all_predictions.extend(predictions)
        
        return np.array(all_predictions)
    
    def save(self, path):
        """
        Save the model to the specified path.
        
        Args:
            path: Path where the model should be saved
        """
        os.makedirs(path, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(path)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(path)
        
        # Save hyperparameters
        with open(os.path.join(path, 'hyperparameters.pt'), 'wb') as f:
            torch.save({
                'max_length': self.max_length,
                'batch_size': self.batch_size,
                'epochs': self.epochs,
                'learning_rate': self.learning_rate
            }, f)
    
    def load(self, path):
        """
        Load the model from the specified path.
        
        Args:
            path: Path from where the model should be loaded
        """
        # Load model
        self.model = DistilBertForSequenceClassification.from_pretrained(path)
        self.model.to(self.device)
        
        # Load tokenizer
        self.tokenizer = DistilBertTokenizer.from_pretrained(path)
        
        # Load hyperparameters
        with open(os.path.join(path, 'hyperparameters.pt'), 'rb') as f:
            hyperparameters = torch.load(f)
            self.max_length = hyperparameters['max_length']
            self.batch_size = hyperparameters['batch_size']
            self.epochs = hyperparameters['epochs']
            self.learning_rate = hyperparameters['learning_rate']
        
        return self