import pandas as pd
from sklearn.model_selection import train_test_split
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os

# Download required NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

def load_data(data_dir='../data', use_predefined_splits=True):
    """
    Load the fake news dataset from TSV files.
    
    Args:
        data_dir: Directory containing the TSV files
        use_predefined_splits: Whether to use the predefined train/test/valid splits
        
    Returns:
        Dictionary containing train, test, and validation splits
    """
    # Define column names for the TSV files
    column_names = [
        'id', 'label', 'statement', 'subject', 'speaker', 'job_title', 'state', 
        'party', 'barely_true_counts', 'false_counts', 'half_true_counts', 
        'mostly_true_counts', 'pants_on_fire_counts', 'context'
    ]
    
    if use_predefined_splits:
        # Load the predefined splits
        train_path = os.path.join(data_dir, 'train.tsv')
        test_path = os.path.join(data_dir, 'test.tsv')
        valid_path = os.path.join(data_dir, 'valid.tsv')
        
        train_df = pd.read_csv(train_path, sep='\t', names=column_names)
        test_df = pd.read_csv(test_path, sep='\t', names=column_names)
        valid_df = pd.read_csv(valid_path, sep='\t', names=column_names)
        
        # Combine all data for full dataset statistics
        full_df = pd.concat([train_df, test_df, valid_df], ignore_index=True)
    else:
        # Load a single file and create splits
        file_path = os.path.join(data_dir, 'train.tsv')  # Default to train.tsv
        full_df = pd.read_csv(file_path, sep='\t', names=column_names)
        
        # Map string labels to binary values for classification
        # Split the dataset
        train_df, temp_df = train_test_split(
            full_df, test_size=0.3, random_state=42, stratify=full_df['label']
        )
        
        test_df, valid_df = train_test_split(
            temp_df, test_size=0.5, random_state=42, stratify=temp_df['label']
        )
    
    # Map the string labels to binary values for classification
    # For this dataset, we'll consider 'true' and 'mostly-true' as REAL (0)
    # and all other labels as FAKE (1)
    label_map = {
        'true': 0, 'mostly-true': 0,  # Real news
        'half-true': 1, 'barely-true': 1, 'false': 1, 'pants-fire': 1  # Fake news
    }
    
    # Apply the label mapping
    train_df['binary_label'] = train_df['label'].map(label_map)
    test_df['binary_label'] = test_df['label'].map(label_map)
    valid_df['binary_label'] = valid_df['label'].map(label_map)
    full_df['binary_label'] = full_df['label'].map(label_map)
    
    return {
        'train': {
            'texts': train_df['statement'],
            'labels': train_df['binary_label'],
            'df': train_df
        },
        'test': {
            'texts': test_df['statement'],
            'labels': test_df['binary_label'],
            'df': test_df
        },
        'valid': {
            'texts': valid_df['statement'],
            'labels': valid_df['binary_label'],
            'df': valid_df
        },
        'full_data': full_df
    }

def preprocess_text(text):
    """
    Preprocess text data for traditional ML models.
    
    Args:
        text: Text to preprocess
        
    Returns:
        Preprocessed text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove punctuation and numbers
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Lemmatization and stopword removal
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    
    return ' '.join(words)