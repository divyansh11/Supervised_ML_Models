import pickle
import numpy as np
import pandas as pd
import streamlit as st
import streamlit_shadcn_ui as ui
import os 
import pygwalker as pyg
from pygwalker.api.streamlit import StreamlitRenderer

script_dir = os.path.dirname(__file__)

# Define image paths
image_path2 = os.path.join(script_dir, 'Images', 'OIP (1).jpg')
image_path3 = os.path.join(script_dir, 'Images', 'OIP.jpg')

st.set_page_config(layout='wide')
# Load images with error handling
try:
    images = [image_path2, image_path3]
    st.image(images, width=200)
except Exception as e:
    st.error(f"An error occurred while loading images: {e}")

# Load model
file_path2 = os.path.join(script_dir, 'model.pkl')
try:
    model = pickle.load(open(file_path2, 'rb'))
except Exception as e:
    st.error(f"An error occurred while loading the model: {e}")

# Prediction dictionary
dict_pred = {1: 'rice', 2: 'maize', 3: 'jute', 4: 'cotton', 5: 'coconut', 6: 'papaya', 
             7: 'orange', 8: 'apple', 9: 'muskmelon', 10: 'watermelon', 11: 'grapes', 
             12: 'mango', 13: 'banana', 14: 'pomegranate', 15: 'lentil', 16: 'blackgram', 
             17: 'mungbean', 18: 'mothbeans', 19: 'pigeonpeas', 20: 'kidneybeans', 
             21: 'chickpea', 22: 'coffee'}

def cropPrediction(N, P, K, temp, humd, ph, rainfall):
    feat = np.array([[N, P, K, temp, humd, ph, rainfall]]).reshape(1, -1)
    try:
        prediction = model.predict(feat)
        if prediction[0] in dict_pred:
            crop = dict_pred[prediction[0]]
            result = f"{crop} is the best crop to be cultivated"
        else:
            result = "Sorry, I don't have an answer for these"
    except Exception as e:
        result = f"An error occurred during prediction: {e}"
    return result

# Creating Streamlit application

st.header("Crop Recommendation Model // Divyansh Sankhla")

n = st.text_input('Enter the value of N')
p = st.text_input('Enter the value of P')
k = st.text_input('Enter the value of K')
temperature = st.text_input('Enter the value of temperature')
humidity = st.text_input('Enter the value of humidity')
ph = st.text_input('Enter the value of ph')
rainfall = st.text_input('Enter the value of rainfall')

file_path1 = os.path.join(script_dir, 'Crop_recommendation.csv')

submit = ui.button("Submit", key="clk_btn")

if submit:
    try:
        # Convert inputs to float
        n = float(n)
        p = float(p)
        k = float(k)
        temperature = float(temperature)
        humidity = float(humidity)
        ph = float(ph)
        rainfall = float(rainfall)
        
        # Generate prediction
        ans = cropPrediction(n, p, k, temperature, humidity, ph, rainfall)
        st.write(ans)
    except ValueError:
        st.write("Please enter valid numeric values for all inputs.")

    st.write("For further analysis kindly refer to the dashboard catalog")

    @st.cache_resource
    def get_pyg_renderer() -> "StreamlitRenderer":
        try:
            df = pd.read_csv(file_path1)
            return StreamlitRenderer(df, spec="./gw_config.json", spec_io_mode="rw")
        except Exception as e:
            st.error(f"An error occurred while loading the CSV file: {e}")
            return None
    
    renderer = get_pyg_renderer()
    
    if renderer:
        renderer.explorer()
