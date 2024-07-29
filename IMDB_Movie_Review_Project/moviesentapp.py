import pickle 
import os
import streamlit as st 
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Initialize the tokenizer
tokenizer = Tokenizer(num_words=4000)
script_dir = os.path.dirname(__file__)

# Define image paths
image_path2 = os.path.join(script_dir, 'Images', 'image2.jpg')
image_path3 = os.path.join(script_dir, 'Images', 'images1.jpg')

st.set_page_config(layout='centered')
st.header("NLP-Driven Sentiment Analysis on IMDB Dataset // Divyansh Sankhla")

try:
    images = [image_path2, image_path3]
    st.image(images, width=200)
except Exception as e:
    st.error(f"An error occurred while loading images: {e}")

file_path_model = os.path.join(script_dir, 'model.pkl')
try:
    with open(file_path_model, 'rb') as model_file:
        model = pickle.load(model_file)
except Exception as e:
    st.error(f"An error occurred while loading the model: {e}")

review = st.text_input("Enter the review of the movie")

# Building a predictive model
def prediction_model(review):
    sequence = tokenizer.texts_to_sequences([review])
    padded_seq = pad_sequences(sequence, maxlen=200)
    prediction = model.predict(padded_seq)
    sentiment = 'positive' if prediction[0][0] > 0.5 else 'negative'
    return sentiment, prediction

submit = st.button("Submit")

if submit:
    if review:
        ans, predict = prediction_model(review)
        st.write("The review is", ans)
        st.write("The prediction score against this is", predict[0][0] * 100, '%')
    else:
        st.error("Please enter a review.")
