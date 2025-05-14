import pandas as pd
from sklearn.model_selection import train_test_split
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

def load_data(file_path, test_size=0.2, random_state=42):
    """
    Load and split the fake news dataset.
    
    Args:
        file_path: Path to the CSV file
        test_size: Proportion of the dataset to include in the test split
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary containing train and test splits
    """
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Ensure the dataset has the expected columns
    required_columns = ['text', 'label']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in the dataset")
    
    # Split the dataset
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df['label']
    )
    
    return {
        'train': {
            'texts': train_df['text'],
            'labels': train_df['label']
        },
        'test': {
            'texts': test_df['text'],
            'labels': test_df['label']
        },
        'full_data': df
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