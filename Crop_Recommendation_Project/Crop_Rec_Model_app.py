import pickle
import numpy as np
import pandas as pd
import streamlit as st
import streamlit_shadcn_ui as ui
import os
import pygwalker as pyg
from pygwalker.api.streamlit import StreamlitRenderer

# Define paths
script_dir = os.path.dirname(__file__)
image_paths = [
    os.path.join(script_dir, 'Images/OIP (1).jpg'),
    os.path.join(script_dir, 'Images/OIP.jpg'),
]
model_path = os.path.join(script_dir, 'model.pkl')
csv_path = os.path.join(script_dir, 'Crop_recommendation.csv')
spec_path = "./gw_config.json"

# Load model
model = pickle.load(open(model_path, 'rb'))

# Define crop prediction function
def cropPrediction(N, P, K, temp, humd, ph, rainfall):
    feat = np.array([[N, P, K, temp, humd, ph, rainfall]]).reshape(1, -1)
    prediction = model.predict(feat)
    return dict_pred.get(prediction[0], "Sorry, I don't have an answer for these")

# Crop dictionary
dict_pred = {
    1: 'rice', 2: 'maize', 3: 'jute', 4: 'cotton', 5: 'coconut', 6: 'papaya', 
    7: 'orange', 8: 'apple', 9: 'muskmelon', 10: 'watermelon', 11: 'grapes', 
    12: 'mango', 13: 'banana', 14: 'pomegranate', 15: 'lentil', 16: 'blackgram', 
    17: 'mungbean', 18: 'mothbeans', 19: 'pigeonpeas', 20: 'kidneybeans', 
    21: 'chickpea', 22: 'coffee'
}

# Streamlit app configuration
st.set_page_config(layout='wide')
st.header("Crop Recommendation Model // Divyansh Sankhla")
st.image(image_paths, width=200)

# Input fields
n = st.text_input('Enter the value of N')
p = st.text_input('Enter the value of P')
k = st.text_input('Enter the value of K')
temperature = st.text_input('Enter the value of temperature')
humidity = st.text_input('Enter the value of humidity')
ph = st.text_input('Enter the value of ph')
rainfall = st.text_input('Enter the value of rainfall')

# Submit button
submit = ui.button("Submit", key="clk_btn")

if submit:
    try:
        # Convert inputs to float
        inputs = [float(x) for x in [n, p, k, temperature, humidity, ph, rainfall]]
        result = cropPrediction(*inputs)
        st.write(result)
    except ValueError:
        st.write("Please enter valid numeric values for all inputs.")

    st.write("For further analysis kindly refer to the dashboard catalog below.")

    @st.cache_resource
    def get_pyg_renderer() -> StreamlitRenderer:
        df = pd.read_csv(csv_path)
        return StreamlitRenderer(df, spec=spec_path, spec_io_mode="rw")
    
    renderer = get_pyg_renderer()
    renderer.explorer()
