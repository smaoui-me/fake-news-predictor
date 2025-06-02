# Fake News Predictor

This project uses Natural Language Processing (NLP) and machine learning to automatically detect whether a news article is real or fake based on its text content. By analyzing the linguistic patterns in news headlines and articles, the model can classify articles as "FAKE" or "REAL" with high accuracy. The goal is to help combat misinformation by providing an automated, scalable tool for fact-checking.

## Dataset

The project now uses the LIAR dataset, a benchmark dataset for fake news detection:

**Citation**: William Yang Wang, "Liar, Liar Pants on Fire": A New Benchmark Dataset for Fake News Detection, Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (ACL 2017).

### Dataset Structure

The dataset is provided in TSV format with the following columns:

1. ID of the statement ([ID].json)
2. Label (pants-fire, false, barely-true, half-true, mostly-true, true)
3. Statement (the text to classify)
4. Subject(s)
5. Speaker
6. Speaker's job title
7. State info
8. Party affiliation
   9-13. Credit history counts (barely true, false, half true, mostly true, pants on fire)
9. Context (venue/location of the speech or statement)

### Binary Classification

For the purpose of this project, we've converted the original 6-class labels into a binary classification:

- **Real News (0)**: Statements labeled as 'true' or 'mostly-true'
- **Fake News (1)**: Statements labeled as 'half-true', 'barely-true', 'false', or 'pants-fire'

### Dataset Distribution

#### Original Label Distribution

| Label       | Count | Percentage |
| ----------- | ----- | ---------- |
| half-true   | 2627  | 20.54%     |
| false       | 2507  | 19.60%     |
| mostly-true | 2454  | 19.19%     |
| barely-true | 2103  | 16.44%     |
| true        | 2053  | 16.05%     |
| pants-fire  | 1047  | 8.19%      |

#### Binary Label Distribution

| Label    | Count | Percentage |
| -------- | ----- | ---------- |
| Fake (1) | 8284  | 64.76%     |
| Real (0) | 4507  | 35.24%     |

#### Dataset Splits

- Training set: 10,240 statements
- Test set: 1,267 statements
- Validation set: 1,284 statements

## Key Findings

- Using the LIAR dataset, a benchmark for fake news detection research
- Binary classification approach (real vs. fake) derived from the original 6-class labels
- Models evaluated on both test and validation sets for robust performance assessment
- Transformer-based models (DistilBERT) show improved performance over traditional ML approaches
- Additional metadata (speaker, context, etc.) provides valuable context for analysis
- Binary classification threshold set at 'mostly-true', with anything less truthful considered fake news

## Project Structure

- **models/** : Contains the model implementations
  - **base_model.py** : Abstract base class defining the model interface
  - **logistic_model.py** : Logistic Regression model with TF-IDF features
  - **distilbert_model.py** : DistilBERT-based transformer model
  - **utils.py** : Utilities for data loading and preprocessing
- **notebooks/** : Contains Jupyter notebooks for analysis
  - **fake_news_analysis.ipynb** : Local notebook for CPU-based processing
  - **fake_news_analysis_colab_tsv.ipynb** : Google Colab notebook for GPU acceleration with TSV dataset
- **data/** : Contains the dataset files
  - **train.tsv** : Training data in TSV format
  - **test.tsv** : Test data in TSV format
  - **valid.tsv** : Validation data in TSV format

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

```bash
pip install torch transformers scikit-learn pandas numpy matplotlib seaborn nltk
```

2. Load the dataset:

```python
from models import load_data

# Load data with predefined train/test/validation splits
data = load_data('data', use_predefined_splits=True)

# Access the different splits
train_texts = data['train']['texts']
train_labels = data['train']['labels']
test_texts = data['test']['texts']
test_labels = data['test']['labels']
valid_texts = data['valid']['texts']
valid_labels = data['valid']['labels']
```

3. Train and evaluate models:

```python
from models import LogisticModel, DistilBertModel

# For Logistic Regression
logistic_model = LogisticModel(max_features=10000, preprocess=True)
logistic_model.train(train_texts, train_labels)

# Evaluate on test set
logistic_metrics = logistic_model.evaluate(test_texts, test_labels)
print(f"Accuracy: {logistic_metrics['accuracy']:.4f}")

# For DistilBERT
distilbert_model = DistilBertModel(max_length=128, batch_size=16, epochs=2)
distilbert_model.train(train_texts, train_labels)

# Evaluate on test set
distilbert_metrics = distilbert_model.evaluate(test_texts, test_labels)
print(f"Accuracy: {distilbert_metrics['accuracy']:.4f}")
```

4. Make predictions on new statements:

```python
statements = [
    "The economy has grown by 4% in the last quarter.",
    "Vaccines contain microchips to track people."
]

# Get predictions (0 for Real, 1 for Fake)
logistic_preds = logistic_model.predict(statements)
distilbert_preds = distilbert_model.predict(statements)

# Map predictions to human-readable labels
label_map = {0: 'Real', 1: 'Fake'}
for i, statement in enumerate(statements):
    print(f"Statement: {statement}")
    print(f"Logistic Regression: {label_map[logistic_preds[i]]}")
    print(f"DistilBERT: {label_map[distilbert_preds[i]]}\n")
```

## Model Performance

Both models are evaluated on various metrics to ensure robust performance:

- Accuracy: How often the model correctly identifies both real and fake news
- Precision: Ability to avoid false positives
- Recall: Ability to identify all fake news articles
- F1-Score: Balanced measure of precision and recall

### Test Set Performance

| Metric    | Logistic Regression | DistilBERT |
| --------- | ------------------- | ---------- |
| Accuracy  | 0.6440              | 0.6803     |
| Precision | 0.6710              | 0.7277     |
| Recall    | 0.8802              | 0.8068     |
| F1-Score  | 0.7615              | 0.7652     |

### Validation Set Performance

| Metric    | Logistic Regression | DistilBERT |
| --------- | ------------------- | ---------- |
| Accuracy  | 0.6760              | 0.6791     |
| Precision | 0.7063              | 0.7374     |
| Recall    | 0.8877              | 0.8125     |
| F1-Score  | 0.7867              | 0.7731     |

### Key Observations

- DistilBERT achieves higher precision and accuracy on both test and validation sets
- Logistic Regression shows stronger recall, identifying more fake news instances
- Both models perform consistently across test and validation sets, indicating good generalization
- The F1-scores are comparable, with Logistic Regression performing slightly better on the validation set

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.
