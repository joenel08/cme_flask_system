# import pickle
# import re
# import nltk
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer

# # ✅ Download NLTK resources
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

# # ✅ Load the model and TF-IDF vectorizer
# # MODEL_PATH = 'model/v1_rf_model.pkl'
# MODEL_PATH = "https://drive.google.com/uc?export=download&id=1ZN05zFYNHKVZKTN0sWDlq-PcPb98m2qd"
# VECTORIZER_PATH = "https://drive.google.com/uc?export=download&id=1ZLQT6vC2WebqFnVleF6Gr6jjVPOTGG96"


# with open(MODEL_PATH, 'rb') as f:
#     loaded_ann_model = pickle.load(f)

# with open(VECTORIZER_PATH, 'rb') as f:
#     loaded_tfidf_vectorizer = pickle.load(f)

# # ✅ Text Cleaning Functions
# def remove_noise(text):
#     """Remove special characters and punctuation."""
#     text = re.sub(r"[^\w\s]", "", text)
#     return text

# def standardize(text):
#     """Convert text to lowercase."""
#     return text.lower()

# def tokenize(text):
#     """Tokenize the text."""
#     tokens = word_tokenize(text)
#     return tokens

# def linguistic_processing(tokens):
#     """Lemmatize tokens and remove stopwords."""
#     lemmatizer = WordNetLemmatizer()
#     tokens = [lemmatizer.lemmatize(token) for token in tokens]

#     stop_words = set(stopwords.words('english'))
#     tokens = [token for token in tokens if token not in stop_words]

#     return tokens

# def preprocess_text(text):
#     """Combine all preprocessing steps into a pipeline."""
#     text = standardize(text)
#     text = remove_noise(text)
#     tokens = tokenize(text)
#     tokens = linguistic_processing(tokens)
#     return ' '.join(tokens)

# # ✅ Sentiment Prediction Function
# def predict_sentiment_label_ann(input_text, model, vectorizer):
#     """
#     Predict sentiment using the pre-trained ANN model.
#     """
#     # Preprocess the input text
#     preprocessed_text = preprocess_text(input_text)

#     # Vectorize the preprocessed text
#     input_text_tfidf = vectorizer.transform([preprocessed_text])

#     # Perform prediction
#     numeric_prediction = model.predict(input_text_tfidf)[0]

#     # Map prediction to labels
#     sentiment_labels = {1: 'Positive', 0: 'Negative', 2: 'Neutral'}
#     return sentiment_labels.get(numeric_prediction, 'Unknown')


# import pickle
# import re
# import nltk
# import requests
# import io
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer

# # ✅ Download NLTK resources
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

# # ✅ Direct download URLs (Google Drive)
# MODEL_URL = "https://drive.google.com/uc?export=download&id=1ZN05zFYNHKVZKTN0sWDlq-PcPb98m2qd"
# VECTORIZER_URL = "https://drive.google.com/uc?export=download&id=1ZLQT6vC2WebqFnVleF6Gr6jjVPOTGG96"

# # ✅ Load model from Google Drive
# model_response = requests.get(MODEL_URL)
# loaded_ann_model = pickle.load(io.BytesIO(model_response.content))

# # ✅ Load vectorizer from Google Drive
# vectorizer_response = requests.get(VECTORIZER_URL)
# loaded_tfidf_vectorizer = pickle.load(io.BytesIO(vectorizer_response.content))

# # ✅ Text Cleaning Functions
# def remove_noise(text):
#     text = re.sub(r"[^\w\s]", "", text)
#     return text

# def standardize(text):
#     return text.lower()

# def tokenize(text):
#     return word_tokenize(text)

# def linguistic_processing(tokens):
#     lemmatizer = WordNetLemmatizer()
#     tokens = [lemmatizer.lemmatize(token) for token in tokens]
#     stop_words = set(stopwords.words('english'))
#     tokens = [token for token in tokens if token not in stop_words]
#     return tokens

# def preprocess_text(text):
#     text = standardize(text)
#     text = remove_noise(text)
#     tokens = tokenize(text)
#     tokens = linguistic_processing(tokens)
#     return ' '.join(tokens)

# # ✅ Prediction Function
# def predict_sentiment_label_ann(input_text, model, vectorizer):
#     preprocessed_text = preprocess_text(input_text)
#     input_text_tfidf = vectorizer.transform([preprocessed_text])
#     numeric_prediction = model.predict(input_text_tfidf)[0]
#     sentiment_labels = {1: 'Positive', 0: 'Negative', 2: 'Neutral'}
#     return sentiment_labels.get(numeric_prediction, 'Unknown')


# # ✅ Example Test
# if __name__ == "__main__":
#     new_text = "The place was absolutely amazing with wonderful staff!"
#     sentiment = predict_sentiment_label_ann(new_text, loaded_ann_model, loaded_tfidf_vectorizer)
#     print(f'Sentiment Prediction: {sentiment}')


# sentiment_analysis.py - Updated version
# sentiment_analysis.py
import pickle
import re
import nltk
import requests
import io
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import gdown

# ✅ Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# ✅ Google Drive file IDs
MODEL_FILE_ID = "1ZN05zFYNHKVZKTN0sWDlq-PcPb98m2qd"
VECTORIZER_FILE_ID = "1ZLQT6vC2WebqFnVleF6Gr6jjVPOTGG96"

# ✅ File paths
MODEL_PATH = "ann_model.pkl"
VECTORIZER_PATH = "tfidf_vectorizer.pkl"

def download_from_gdrive(file_id, destination):
    """Download file from Google Drive if it doesn't exist"""
    # Check if file already exists
    if os.path.exists(destination):
        print(f"File already exists: {destination}")
        return destination
    
    print(f"Downloading {destination}...")
    url = f"https://drive.google.com/uc?id={file_id}"
    try:
        gdown.download(url, destination, quiet=False)
        print(f"Downloaded: {destination}")
        return destination
    except Exception as e:
        print(f"Error downloading {destination}: {e}")
        raise

def download_file_from_google_drive(file_id):
    """Manual download method as fallback"""
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    
    response = session.get(URL, params={'id': file_id}, stream=True)
    token = None
    
    # Handle confirmation for large files
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
            break
    
    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
    
    return io.BytesIO(response.content)

def load_model_and_vectorizer():
    """Load or download model and vectorizer"""
    global loaded_ann_model, loaded_tfidf_vectorizer
    
    try:
        # Try to load existing files first
        if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
            print("Loading existing model files...")
            with open(MODEL_PATH, 'rb') as f:
                loaded_ann_model = pickle.load(f)
            with open(VECTORIZER_PATH, 'rb') as f:
                loaded_tfidf_vectorizer = pickle.load(f)
            print("Model files loaded successfully!")
            return loaded_ann_model, loaded_tfidf_vectorizer
        
        # If files don't exist, download them
        print("Model files not found. Downloading...")
        
        # Option 1: Using gdown library
        try:
            download_from_gdrive(MODEL_FILE_ID, MODEL_PATH)
            download_from_gdrive(VECTORIZER_FILE_ID, VECTORIZER_PATH)
            
            # Load the downloaded files
            with open(MODEL_PATH, 'rb') as f:
                loaded_ann_model = pickle.load(f)
            with open(VECTORIZER_PATH, 'rb') as f:
                loaded_tfidf_vectorizer = pickle.load(f)
                
            print("Model files downloaded and loaded successfully!")
            return loaded_ann_model, loaded_tfidf_vectorizer
            
        except Exception as e:
            print(f"Error with gdown: {e}")
            print("Trying manual download method...")
            
            # Option 2: Manual download method
            model_content = download_file_from_google_drive(MODEL_FILE_ID)
            loaded_ann_model = pickle.load(model_content)
            
            vectorizer_content = download_file_from_google_drive(VECTORIZER_FILE_ID)
            loaded_tfidf_vectorizer = pickle.load(vectorizer_content)
            
            # Save downloaded files for future use
            with open(MODEL_PATH, 'wb') as f:
                pickle.dump(loaded_ann_model, f)
            with open(VECTORIZER_PATH, 'wb') as f:
                pickle.dump(loaded_tfidf_vectorizer, f)
                
            print("Model files downloaded (manual) and saved for future use!")
            return loaded_ann_model, loaded_tfidf_vectorizer
            
    except Exception as e:
        print(f"Error loading model files: {e}")
        raise

# ✅ Initialize model and vectorizer
try:
    loaded_ann_model, loaded_tfidf_vectorizer = load_model_and_vectorizer()
except Exception as e:
    print(f"Failed to initialize model: {e}")
    loaded_ann_model = None
    loaded_tfidf_vectorizer = None

# ✅ Text Cleaning Functions
def remove_noise(text):
    text = re.sub(r"[^\w\s]", "", text)
    return text

def standardize(text):
    return text.lower()

def tokenize(text):
    return word_tokenize(text)

def linguistic_processing(tokens):
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    return tokens

def preprocess_text(text):
    text = standardize(text)
    text = remove_noise(text)
    tokens = tokenize(text)
    tokens = linguistic_processing(tokens)
    return ' '.join(tokens)

# ✅ Prediction Function
def predict_sentiment_label_ann(input_text, model=None, vectorizer=None):
    # Use provided model/vectorizer or global ones
    if model is None:
        model = loaded_ann_model
    if vectorizer is None:
        vectorizer = loaded_tfidf_vectorizer
    
    if model is None or vectorizer is None:
        raise ValueError("Model or vectorizer not initialized")
    
    preprocessed_text = preprocess_text(input_text)
    input_text_tfidf = vectorizer.transform([preprocessed_text])
    numeric_prediction = model.predict(input_text_tfidf)[0]
    sentiment_labels = {1: 'Positive', 0: 'Negative', 2: 'Neutral'}
    return sentiment_labels.get(numeric_prediction, 'Unknown')