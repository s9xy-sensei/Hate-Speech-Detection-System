import streamlit as st
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.text import Tokenizer
import string
from nltk.tokenize import RegexpTokenizer
import pandas as pd
import pickle
import nltk
nltk.download('stopwords')
nltk.download('wordnet')

# Load pre-trained model (replace with your model path)
loaded_model = load_model('lstm_model.h5')
tok = Tokenizer(num_words=2000)
max_len = 500
with open('tokenizer_f.pickle', 'rb') as handle:
    tok1 = pickle.load(handle)

# Preprocessing functions (assuming these are defined elsewhere)
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove email addresses
    text = re.sub('@[^\s]+', ' ', text)
    # Remove URLs
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', ' ', text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove numbers
    text = re.sub('[0-9]+', '', text)
    # Tokenize
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Function to make prediction
def predict(text):
    preprocessed_text = preprocess_text(text)
    sequence = tok1.texts_to_sequences([preprocessed_text])
    padded_sequence = pad_sequences(sequence, maxlen=max_len)
    prediction = loaded_model.predict(padded_sequence)
    return prediction

# Streamlit app
st.title("Hate Speech Detection App")
st.subheader("Enter tweet for Hate Speech Detection:")
user_input = st.text_input("Enter tweet to analyze:")
if st.button("Submit Your Tweet"):
    prediction = predict(user_input)
    if prediction > 0.5:
        st.success('No Hate Speech')
    else:
        st.error('Hate Speech')
