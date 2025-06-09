import pickle
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ✅ Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# ✅ Load the model and TF-IDF vectorizer
MODEL_PATH = 'model/v1_rf_model.pkl'
VECTORIZER_PATH = 'model/v1_tfidf_vectorizer.pkl'

with open(MODEL_PATH, 'rb') as f:
    loaded_ann_model = pickle.load(f)

with open(VECTORIZER_PATH, 'rb') as f:
    loaded_tfidf_vectorizer = pickle.load(f)

# ✅ Text Cleaning Functions
def remove_noise(text):
    """Remove special characters and punctuation."""
    text = re.sub(r"[^\w\s]", "", text)
    return text

def standardize(text):
    """Convert text to lowercase."""
    return text.lower()

def tokenize(text):
    """Tokenize the text."""
    tokens = word_tokenize(text)
    return tokens

def linguistic_processing(tokens):
    """Lemmatize tokens and remove stopwords."""
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    return tokens

def preprocess_text(text):
    """Combine all preprocessing steps into a pipeline."""
    text = standardize(text)
    text = remove_noise(text)
    tokens = tokenize(text)
    tokens = linguistic_processing(tokens)
    return ' '.join(tokens)

# ✅ Sentiment Prediction Function
def predict_sentiment_label_ann(input_text, model, vectorizer):
    """
    Predict sentiment using the pre-trained ANN model.
    """
    # Preprocess the input text
    preprocessed_text = preprocess_text(input_text)

    # Vectorize the preprocessed text
    input_text_tfidf = vectorizer.transform([preprocessed_text])

    # Perform prediction
    numeric_prediction = model.predict(input_text_tfidf)[0]

    # Map prediction to labels
    sentiment_labels = {1: 'Positive', 0: 'Negative', 2: 'Neutral'}
    return sentiment_labels.get(numeric_prediction, 'Unknown')


# # ✅ Example Test
# if __name__ == "__main__":
#     new_text = "The place was absolutely amazing with wonderful staff!"
#     sentiment = predict_sentiment_label_ann(new_text, loaded_ann_model, loaded_tfidf_vectorizer)
#     print(f'Sentiment Prediction: {sentiment}')
