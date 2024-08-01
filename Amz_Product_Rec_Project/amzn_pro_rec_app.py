import pandas as pd
import numpy as np
import nltk
nltk.download('punkt')
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from PIL import Image
<<<<<<< HEAD
import os

# Download the 'punkt' tokenizer model if not already available
nltk.download('punkt')

# File paths
=======
import os 

# To read the data
>>>>>>> refs/remotes/origin/master
script_dir = os.path.dirname(__file__)
file_path = os.path.join(script_dir, 'amazon_product.csv')
file_path2 = os.path.join(script_dir, 'amazon_rec.png')

# Read the data
df = pd.read_csv(file_path)

# Drop unnecessary column
df.drop('id', axis=1, inplace=True)

# Initialize stemmer and vectorizer
stemmer = SnowballStemmer('english')
tfid = TfidfVectorizer(tokenizer=lambda text: tokenize_stem(text).split())

# Tokenizer and stemmer function
def tokenize_stem(text):
    tokens = nltk.word_tokenize(text.lower())
    stemmed = [stemmer.stem(w) for w in tokens]
    return " ".join(stemmed)

# Apply stemming to the DataFrame
df['stemmed_tokens'] = df.apply(lambda row: tokenize_stem(row['Title'] + " " + row['Description']), axis=1)

# Function for cosine similarity
def cosine_sim(text1, text2):
    matrix = tfid.fit_transform([text1, text2])
    similarity_matrix = cosine_similarity(matrix)
    return similarity_matrix[0, 1]

# Function to search for similar products
def search_product(query):
    stemmed_query = tokenize_stem(query)
    df['similarity'] = df['stemmed_tokens'].apply(lambda x: cosine_sim(stemmed_query, x))
    res = df.sort_values(by=['similarity'], ascending=False).head(15)[['Title', 'Description', 'Category']]
    return res

# Streamlit app setup
st.set_page_config(layout='centered')
img = Image.open(file_path2)
st.image(img, width=400)

st.header('Scalable Amazon Recommendations via Matrix Factorization and NLTK Text Analysis // Divyansh Sankhla')

# User input
quest = st.text_input('Enter the name of product')
submit = st.button('Search')

if submit:
    ans = search_product(quest)
    st.write(ans)
