# Fake News Predictor

This project uses Natural Language Processing (NLP) and machine learning to automatically detect whether a news article is real or fake based on its text content. By analyzing the linguistic patterns in news headlines and articles, the model can classify articles as "FAKE" or "REAL" with high accuracy. The goal is to help combat misinformation by providing an automated, scalable tool for fact-checking.

## Dataset

The dataset used for this project is the "Fake News" dataset from Kaggle (link: https://www.kaggle.com/datasets/khushikyad001/fake-news-detection). It contains 4,000 news articles, labeled as "FAKE" or "REAL".

## Key Findings

- Dataset of 4,000 articles is well-balanced (50.65% fake, 49.35% real)
- Despite optimal class balance, models perform at random chance level (~50% accuracy)
- Limited dataset size is a critical factor affecting model performance
- Larger datasets (>20,000 articles) likely needed for effective detection
- Current dataset is significantly smaller than industry benchmarks:
  - LIAR dataset: ~12,800 statements
  - FakeNewsNet: ~23,000 articles
  - ISOT Fake News dataset: ~44,000 articles

## Project Structure

- models/ : Contains the model implementations
- base_model.py : Abstract base class defining the model interface
- logistic_model.py : Logistic Regression model with TF-IDF features
- distilbert_model.py : DistilBERT-based transformer model
- notebooks/ : Contains Jupyter notebooks for analysis
- fake_news_analysis.ipynb : Local notebook for CPU-based processing
- fake_news_analysis_colab.ipynb : Google Colab notebook for GPU acceleration
- data/ : Contains the dataset files

## Models

### 1. Logistic Regression Model

- Uses TF-IDF vectorization for feature extraction
- Includes text preprocessing pipeline
- Lightweight and fast for inference
- Good baseline performance

### 2. DistilBERT Model

- Based on the DistilBERT transformer architecture
- Fine-tuned for fake news classification
- More sophisticated text understanding
- Better performance but requires more computational resources

## Features

- Text preprocessing and cleaning
- Feature extraction using TF-IDF and transformers
- Model evaluation with multiple metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
- Confusion matrix visualization
- Support for both traditional ML and deep learning approaches

## Requirements

- Python 3.x
- PyTorch
- Transformers (Hugging Face)
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn

## Usage

1. Install dependencies:

pip install torch transformers scikit-learn pandas numpy matplotlib seaborn

2. Train and evaluate models:

from models import LogisticModel, DistilBertModel

# For Logistic Regression

logistic_model = LogisticModel()
logistic_model.train(texts, labels)
predictions = logistic_model.predict(test_texts)

# For DistilBERT

distilbert_model = DistilBertModel()
distilbert_model.train(texts, labels)
predictions = distilbert_model.predict(test_texts)

## Model Performance

Both models are evaluated on various metrics to ensure robust performance:

- Accuracy: How often the model correctly identifies both real and fake news
- Precision: Ability to avoid false positives
- Recall: Ability to identify all fake news articles
- F1-Score: Balanced measure of precision and recall

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.
