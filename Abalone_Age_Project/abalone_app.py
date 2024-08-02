import streamlit as st
import pickle
import pandas as pd 
import numpy as np 
import os 
from PIL import Image

# Directory of files
script_dir = os.path.dirname(__file__)
file_path = os.path.join(script_dir, 'model.pkl')
img_path=os.path.join(script_dir,'abalone.jpg')

model=pickle.load(open(file_path,'rb'))

st.set_page_config(layout='centered')
st.image(image=img_path,width=200)
st.header('Abalone Chrono-Prediction Using Regression Models')

# Prediction Model

def prediction(s,l,d,h,ww,sw,vw,shw):
    feat=np.array([[s,l,d,h,ww,sw,vw,shw]])
    predict=model.predict(feat).reshape(1,-1)
    return predict[0]

s=st.text_input('Enter the value of sex, mention M, F,I')
if(s=='M'):
    s=0 
elif(s=='F'):
    s=1
else:
    s=2
l=st.text_input('Enter the value of length')
d=st.text_input('Enter the value of diameter')
h=st.text_input('Enter the value of height')
ww=st.text_input('Enter the value for whole weight')
sw=st.text_input('Enter the value for shell weight')
vw=st.text_input('Enter the value for Viscera weight')
shw=st.text_input('Enter the value for Shell Weight')

submit=st.button('Submit')

if submit:
    try:
        l = float(l)
        d = float(d)
        h = float(h)
        ww = float(ww)
        sw = float(sw)
        vw = float(vw)
        shw = float(shw)
        ans=prediction(s,l,d,h,ww,sw,vw,shw)
        st.write("The age of your abalone is",ans[0])
    except ValueError:
        st.error("Please enter valid numerical values")


    
    

