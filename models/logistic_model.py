import pickle
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .base_model import BaseModel
from .utils import preprocess_text

class LogisticModel(BaseModel):
    """
    Logistic Regression model with TF-IDF for fake news classification.
    """
    
    def __init__(self, max_features=10000, preprocess=True):
        """
        Initialize the model.
        
        Args:
            max_features: Maximum number of features for TF-IDF
            preprocess: Whether to preprocess the text data
        """
        self.max_features = max_features
        self.preprocess = preprocess
        
        # Create the pipeline
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=max_features)),
            ('classifier', LogisticRegression(random_state=42, max_iter=1000))
        ])
        
    def _preprocess_data(self, texts):
        """
        Preprocess the text data if required.
        
        Args:
            texts: List or Series of text data
            
        Returns:
            Preprocessed texts
        """
        if not self.preprocess:
            return texts
        
        return [preprocess_text(text) for text in texts]
    
    def train(self, texts, labels):
        """
        Train the model on the provided texts and labels.
        
        Args:
            texts: List or Series of text data
            labels: List or Series of labels (0 for real, 1 for fake)
        """
        processed_texts = self._preprocess_data(texts)
        self.pipeline.fit(processed_texts, labels)
        
        return self
    
    def predict(self, texts):
        """
        Make predictions on the provided texts.
        
        Args:
            texts: List or Series of text data
            
        Returns:
            numpy array of predictions (0 for real, 1 for fake)
        """
        processed_texts = self._preprocess_data(texts)
        return self.pipeline.predict(processed_texts)
    
    def predict_proba(self, texts):
        """
        Get prediction probabilities for the provided texts.
        
        Args:
            texts: List or Series of text data
            
        Returns:
            numpy array of prediction probabilities
        """
        processed_texts = self._preprocess_data(texts)
        return self.pipeline.predict_proba(processed_texts)
    
    def save(self, path):
        """
        Save the model to the specified path.
        
        Args:
            path: Path where the model should be saved
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'pipeline': self.pipeline,
                'max_features': self.max_features,
                'preprocess': self.preprocess
            }, f)
    
    def load(self, path):
        """
        Load the model from the specified path.
        
        Args:
            path: Path from where the model should be loaded
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)
            
        self.pipeline = data['pipeline']
        self.max_features = data['max_features']
        self.preprocess = data['preprocess']
        
        return self