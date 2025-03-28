import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
from sklearn.decomposition import PCA
from tqdm import tqdm  # Fixed tqdm import
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Download NLTK resources (only do this once)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# Load dataset
data = "VOC_DATA.csv"
df = pd.read_csv(data, encoding='latin1')
df = df.copy()  # Avoid SettingWithCopyWarning
print(df.head())

# Check for missing and duplicate values
sum_na = df.isna().sum()
sum_duplicate = df.duplicated().sum()
data_size = df.shape

print(f"Sum of NaN values in each column:\n{sum_na}")
print(f"\nSum of duplicate rows: {sum_duplicate}")
print(f"\nData size: {data_size}")

# Drop NaN values only from `Feedback` column (not entire DataFrame)
df = df.dropna(subset=['Feedback']).drop_duplicates()
latest_size = df.shape

print(f"\nLatest Data size: {latest_size}")

"""# Data Pipeline"""

# Function to remove noise characters
def Final_remove_noise(text):
    text = str(text)  # Ensure input is a string
    return re.sub(r"[^\w\s]", " ", text)

# Function to tokenize the text
def Final_tokenize(text):
    return word_tokenize(text)

# Function to normalize characters
def Final_normalize_characters(tokens):
    return [token.lower() for token in tokens]

# Function to perform POS tagging
def Final_pos_tagging(tokens):
    return pos_tag(tokens)

# Function to perform linguistic processing
def Final_linguistic_processing(tokens):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    return [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]

# Function to convert token list back to string
def Final_list_to_string(tokens):
    return ' '.join(tokens)

# Main function to clean the text
def clean_text(text):
    text = Final_remove_noise(text)
    tokens = Final_tokenize(text)
    tokens = Final_normalize_characters(tokens)
    Final_pos_tagging(tokens)  # Optional POS tagging
    tokens = Final_linguistic_processing(tokens)
    return Final_list_to_string(tokens)

# Apply clean_text function with tqdm progress bar
tqdm.pandas()
df['Feedback'] = df['Feedback'].fillna("")  # Ensure no NaN values in Feedback
df['Cleaned Feedback'] = df['Feedback'].progress_apply(clean_text)

# Check Department distribution
dep_count = df['Department'].value_counts()
print(dep_count)

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Function to analyze sentiment and add new columns
def analyze_sentiment(text):
    scores = analyzer.polarity_scores(text)
    return scores['compound']

# Apply sentiment analysis with tqdm progress bar
df['Sentiment Compound'] = df['Cleaned Feedback'].progress_apply(analyze_sentiment)

# Label feedback based on compound score
def label_sentiment(compound_score):
    if compound_score >= 0.05:
        return 'Positive'
    elif compound_score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

df['Sentiment Label'] = df['Sentiment Compound'].apply(label_sentiment)

# Print Sentiment distribution
print(df['Sentiment Label'].value_counts())

# Display cleaned data
df.head()
