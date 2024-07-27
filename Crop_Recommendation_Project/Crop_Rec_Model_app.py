import pickle
import numpy as np
import pandas as pd
import sklearn 
import streamlit as st
import streamlit_shadcn_ui as ui
import os 
import pygwalker as pyg
from pygwalker.api.streamlit import StreamlitRenderer

script_dir = os.path.dirname(__file__)

image_path2 = os.path.join(script_dir, 'Images/OIP (1).jpg')
image_path3 = os.path.join(script_dir, 'Images/OIP.jpg')
image_path4 = os.path.join(script_dir, 'images/R.jpg')

images = [
    image_path2,image_path3,image_path4
]

file_path2=os.path.join(script_dir, 'model.pkl')
model=pickle.load(open(file_path2,'rb'))


dict_pred= {1: 'rice',
 2: 'maize',
 3: 'jute',
 4: 'cotton',
 5: 'coconut',
 6: 'papaya',
 7: 'orange',
 8: 'apple',
 9: 'muskmelon',
 10: 'watermelon',
 11: 'grapes',
 12: 'mango',
 13: 'banana',
 14: 'pomegranate',
 15: 'lentil',
 16: 'blackgram',
 17: 'mungbean',
 18: 'mothbeans',
 19: 'pigeonpeas',
 20: 'kidneybeans',
 21: 'chickpea',
 22: 'coffee'}

def cropPrediction(N,P,K,temp,humd,ph,rainfall):
    feat=np.array([[N,P,K,temp,humd,ph,rainfall]]).reshape(1,-1)
    prediction=model.predict(feat)

    if prediction[0] in dict_pred:
        crop=dict_pred[prediction[0]]
        result=f"{crop} is the best crop to be culticated"
    else:
        result="Sorry, I don't have answer for these"
    return result

# creating streamlit application
st.set_page_config(layout='wide')
st.header("Crop Recommendation Model // Divyansh Sankhla")
st.image(images, width=200)
n=st.text_input('Enter the value of N')
p	=st.text_input('Enter the value of P')
k	=st.text_input('Enter the value of K')
temperature	=st.text_input('Enter the value of temperature')
humidity	=st.text_input('Enter the value of humidity')
ph	=st.text_input('Enter the value of ph')
rainfall=st.text_input('Enter the value of rainfall')


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

    @st.cache_resource
    def get_pyg_renderer() -> "StreamlitRenderer":
        df = pd.read_csv(file_path1)
        # If you want to use feature of saving chart config, set `spec_io_mode="rw"`
        return StreamlitRenderer(df, spec="./gw_config.json", spec_io_mode="rw")
    
    renderer = get_pyg_renderer()
    
    renderer.explorer()



    
