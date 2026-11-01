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

ranges={'height(cm)':(140,195),'Gtp':(5,250),'hemoglobin':(8,20),'triglyceride':(50,500),'age':(20,80),'weight(kg)':(40,130),'waist(cm)':(55,130),'HDL':(20,100),'serum creatinine':(0.4,2.0),'ALT':(5,150),'fasting blood sugar':(60,250),'relaxation':(40,120),'LDL':(40,250),'systolic':(85,210)}

with st.form('prediction'):
    c1,c2,c3=st.columns(3)
    with c1:
        gender=st.selectbox('Gender', ['F','M'], index=['F','M'].index(vals['gender']))
        age=st.number_input('Age (years)',*ranges['age'],value=int(vals['age']),step=1)
        height=st.number_input('Height (cm)',*ranges['height(cm)'],value=float(vals['height(cm)']),step=1.0)
        weight=st.number_input('Weight (kg)',*ranges['weight(kg)'],value=float(vals['weight(kg)']),step=.5)
        waist=st.number_input('Waist (cm)',*ranges['waist(cm)'],value=float(vals['waist(cm)']),step=.5)
    with c2:
        systolic=st.number_input('Systolic BP',*ranges['systolic'],value=int(vals['systolic']),step=1)
        relaxation=st.number_input('Relaxation / diastolic BP',*ranges['relaxation'],value=int(vals['relaxation']),step=1)
        fbs=st.number_input('Fasting blood sugar',*ranges['fasting blood sugar'],value=float(vals['fasting blood sugar']),step=1.0)
        trig=st.number_input('Triglyceride',*ranges['triglyceride'],value=float(vals['triglyceride']),step=1.0)
        hdl=st.number_input('HDL',*ranges['HDL'],value=float(vals['HDL']),step=.5)
    with c3:
        ldl=st.number_input('LDL',*ranges['LDL'],value=float(vals['LDL']),step=1.0)
        hb=st.number_input('Hemoglobin',*ranges['hemoglobin'],value=float(vals['hemoglobin']),step=.1)
        creat=st.number_input('Serum creatinine',*ranges['serum creatinine'],value=float(vals['serum creatinine']),step=.01)
        alt=st.number_input('ALT',*ranges['ALT'],value=float(vals['ALT']),step=1.0)
        gtp=st.number_input('Gtp',*ranges['Gtp'],value=float(vals['Gtp']),step=1.0)
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
