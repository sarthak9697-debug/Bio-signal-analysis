import streamlit as st, pandas as pd, pickle, numpy as np
from pathlib import Path

st.set_page_config(page_title='Bio Signal Analysis | Smoking Risk', page_icon='🫀', layout='wide')
BASE=Path(__file__).resolve().parent
bundle=pickle.load(open(BASE/'model.pkl','rb')); model=bundle['model']; scaler=bundle['scaler']; features=bundle['features']

st.title('🫀 Bio Signal Analysis — Smoking Risk Classifier')
st.caption('Recruiter-friendly ML demonstration using routine biometric and blood-test signals. This is a portfolio demonstration, not a medical diagnostic tool.')

samples={
 'Typical lower-risk profile': {'gender':'F','height(cm)':160,'Gtp':18,'hemoglobin':13.2,'triglyceride':105,'age':28,'weight(kg)':55,'waist(cm)':70,'HDL':65,'serum creatinine':0.7,'ALT':16,'fasting blood sugar':88,'relaxation':68,'LDL':92,'systolic':112},
 'Typical higher-risk profile': {'gender':'M','height(cm)':172,'Gtp':70,'hemoglobin':16.2,'triglyceride':220,'age':52,'weight(kg)':88,'waist(cm)':105,'HDL':38,'serum creatinine':1.1,'ALT':55,'fasting blood sugar':126,'relaxation':82,'LDL':165,'systolic':148}
}
choice=st.selectbox('Load a built-in sample profile', ['Custom']+list(samples))
vals=samples.get(choice, samples['Typical lower-risk profile'])

ranges={'height(cm)':(140,195,1.0),'Gtp':(5,250,1.0),'hemoglobin':(8,20,0.1),'triglyceride':(50,500,1.0),'age':(20,80,1),'weight(kg)':(40,130,0.5),'waist(cm)':(55,130,0.5),'HDL':(20,100,0.5),'serum creatinine':(0.4,2.0,0.01),'ALT':(5,150,1.0),'fasting blood sugar':(60,250,1.0),'relaxation':(40,120,1),'LDL':(40,250,1.0),'systolic':(85,210,1)}

# Keep min, max, default value, and step consistently typed to avoid
# StreamlitMixedNumericTypesError. The model accepts numeric floats.
def number_field(label, feature):
    minimum, maximum, step = ranges[feature]
    return st.number_input(
        label,
        min_value=float(minimum),
        max_value=float(maximum),
        value=float(vals[feature]),
        step=float(step),
    )

with st.form('prediction'):
    c1,c2,c3=st.columns(3)
    with c1:
        gender=st.selectbox('Gender', ['F','M'], index=['F','M'].index(vals['gender']))
        age=number_field('Age (years)', 'age')
        height=number_field('Height (cm)', 'height(cm)')
        weight=number_field('Weight (kg)', 'weight(kg)')
        waist=number_field('Waist (cm)', 'waist(cm)')
    with c2:
        systolic=number_field('Systolic BP', 'systolic')
        relaxation=number_field('Relaxation / diastolic BP', 'relaxation')
        fbs=number_field('Fasting blood sugar', 'fasting blood sugar')
        trig=number_field('Triglyceride', 'triglyceride')
        hdl=number_field('HDL', 'HDL')
    with c3:
        ldl=number_field('LDL', 'LDL')
        hb=number_field('Hemoglobin', 'hemoglobin')
        creat=number_field('Serum creatinine', 'serum creatinine')
        alt=number_field('ALT', 'ALT')
        gtp=number_field('Gtp', 'Gtp')
    submitted=st.form_submit_button('🔎 Analyze Bio Signals', use_container_width=True)

if submitted:
    row=pd.DataFrame([[0 if gender=='F' else 1,height,gtp,hb,trig,age,weight,waist,hdl,creat,alt,fbs,relaxation,ldl,systolic]],columns=features)
    pred=int(model.predict(scaler.transform(row))[0]); proba=float(model.predict_proba(scaler.transform(row))[0][1]) if hasattr(model,'predict_proba') else None
    st.subheader('Prediction Result')
    if pred==1: st.error('Model prediction: SMOKER / HIGHER SMOKING-RISK CLASS')
    else: st.success('Model prediction: NON-SMOKER / LOWER SMOKING-RISK CLASS')
    if proba is not None: st.metric('Estimated model probability of smoking',f'{proba:.1%}')
    st.info('Interpretation: this probability is a machine-learning output, not a clinical diagnosis. Use the model only as a portfolio/demo system.')

with st.expander('What does this project demonstrate?'):
    st.write('EDA, feature selection, classification, model comparison, preprocessing, model serialization and deployment-ready inference with 15 input signals.')
