from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

class BaseModel(ABC):
    """
    Abstract base class for fake news classification models.
    All model implementations should inherit from this class.
    """
    
    @abstractmethod
    def train(self, texts, labels):
        """
        Train the model on the provided texts and labels.
        
        Args:
            texts: List or Series of text data
            labels: List or Series of labels (0 for real, 1 for fake)
        """
        pass
    
    @abstractmethod
    def predict(self, texts):
        """
        Make predictions on the provided texts.
        
        Args:
            texts: List or Series of text data
            
        Returns:
            numpy array of predictions (0 for real, 1 for fake)
        """
        pass
    
    @abstractmethod
    def save(self, path):
        """
        Save the model to the specified path.
        
        Args:
            path: Path where the model should be saved
        """
        pass
    
    @abstractmethod
    def load(self, path):
        """
        Load the model from the specified path.
        
        Args:
            path: Path from where the model should be loaded
        """
        pass
    
    def evaluate(self, texts, labels):
        """
        Evaluate the model on the provided texts and labels.
        
        Args:
            texts: List or Series of text data
            labels: List or Series of labels (0 for real, 1 for fake)
            
        Returns:
            Dictionary containing evaluation metrics
        """
        predictions = self.predict(texts)
        
        # Handle string labels
        unique_labels = sorted(set(labels))
        if len(unique_labels) == 2 and all(isinstance(label, str) for label in unique_labels):
            # For string labels like 'Real' and 'Fake'
            pos_label = 'Fake'  # Assuming 'Fake' is the positive class
            metrics = {
                'accuracy': accuracy_score(labels, predictions),
                'precision': precision_score(labels, predictions, pos_label=pos_label),
                'recall': recall_score(labels, predictions, pos_label=pos_label),
                'f1_score': f1_score(labels, predictions, pos_label=pos_label),
                'confusion_matrix': confusion_matrix(labels, predictions).tolist()
            }
        else:
            # For numeric labels (0, 1)
            metrics = {
                'accuracy': accuracy_score(labels, predictions),
                'precision': precision_score(labels, predictions),
                'recall': recall_score(labels, predictions),
                'f1_score': f1_score(labels, predictions),
                'confusion_matrix': confusion_matrix(labels, predictions).tolist()
            }
        
        return metrics