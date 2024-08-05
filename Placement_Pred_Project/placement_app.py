import pandas as pd 
import numpy as np 
import pickle
import streamlit as st 
import os 

script_dir = os.path.dirname(__file__)
file_path=os.path.join(script_dir,'model.pkl')

model=pickle.load(open(file_path,'rb'))

st.set_page_config(layout='centered')
st.header('MLRecruiter: Data-Driven Talent Acquisition using GBC')

def prediction_model(ssc_p,hsc_p,degree_p,etest_p,mba_p,salary,gender,ssc,hsc,hsc_s,degree,workex,spec):
    ssc_per=ssc_p
    hsc_per=hsc_p
    degree_per=degree_p
    etest_per=etest_p
    mba_per=mba_p
    sal=salary
    gen=1 if gender=='male' else 0
    ssc=1 if ssc=='others' else 0 
    hsc=1 if hsc=='others' else 0
    hsc_comm=1 if hsc_s=='Commerce' else 0
    hsc_s_Science=1 if hsc_s=='science' else 0
    degree_t_Others=1 if degree=='others' else 0
    degree_t_SciTech=1 if degree=='Sci&Tech' else 0
    workex=1 if workex=='Yes' else 0
    specialisation_MktHR=1 if spec=='Mkt&HR' else 0

    data=np.array([[ssc_per,hsc_per,degree_per,etest_per,mba_per,sal,gen,ssc,hsc,hsc_comm,hsc_s_Science,degree_t_Others,degree_t_SciTech,workex,specialisation_MktHR]]).reshape(1, -1)
    try:
        prediction = model.predict(data)
        if prediction[0]==1:
            status='placed'
        else:
            status='not placed'

        result=f'The candidate would be {status}'
    except Exception as e:
        result = f"An error occurred during prediction: {e}"
    return result



ssc_p=st.text_input('Enter value for SSC Percentage')
hsc_p=st.text_input('Enter the valye for HSC Percentage')
degree_p=st.text_input('Enter the value of percentage obtained in degree')
etest_p=st.text_input('Enter the value for percentage scored in e_test')
mba_p=st.text_input('Enter the value for percentage scored in MBA')
salary=st.text_input('what is your salary')
gender=st.selectbox('what is the gender',('male','female'))
ssc=st.selectbox('what was your ssc',('central','others'))
hsc=st.selectbox('what was your hsc',('central','others'))
hsc_s=st.selectbox('what was your hsc stream',('Commerce','science','arts'))
degree=st.selectbox('what was your degree',('Comm&Mgmt','Sci&Tech','Others'))
workex=st.selectbox('Do you have workex',('Yes','No'))
spec=st.selectbox('what is your specialization',('Mkt&Fin','Mkt&HR'))


submit=st.button('Submit')

if submit:
    ssc_p=float(ssc_p)
    hsc_p=float(hsc_p)
    degree_p=float(degree_p)
    etest_p=float(etest_p)
    mba_p=float(mba_p)
    salary=float(salary)
    ans=prediction_model(ssc_p,hsc_p,degree_p,etest_p,mba_p,salary,gender,ssc,hsc,hsc_s,degree,workex,spec)
    st.write(ans)
    