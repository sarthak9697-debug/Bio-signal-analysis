from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pathlib import Path
import pickle, pandas as pd

BASE=Path(__file__).resolve().parent.parent
with open(BASE/'model.pkl','rb') as f: bundle=pickle.load(f)
model,scaler,features=bundle['model'],bundle['scaler'],bundle['features']
app=FastAPI(title='Bio Signal Analysis API',version='1.0.0')

class Patient(BaseModel):
    gender:str=Field(...,description='F or M')
    height_cm:float=Field(...,ge=140,le=195)
    gtp:float=Field(...,ge=5,le=250)
    hemoglobin:float=Field(...,ge=8,le=20)
    triglyceride:float=Field(...,ge=50,le=500)
    age:int=Field(...,ge=20,le=80)
    weight_kg:float=Field(...,ge=40,le=130)
    waist_cm:float=Field(...,ge=55,le=130)
    hdl:float=Field(...,ge=20,le=100)
    serum_creatinine:float=Field(...,ge=.4,le=2.0)
    alt:float=Field(...,ge=5,le=150)
    fasting_blood_sugar:float=Field(...,ge=60,le=250)
    relaxation:int=Field(...,ge=40,le=120)
    ldl:float=Field(...,ge=40,le=250)
    systolic:int=Field(...,ge=85,le=210)

def frame(p):
    if p.gender.upper() not in {'F','M'}: raise HTTPException(400,'gender must be F or M')
    return pd.DataFrame([[0 if p.gender.upper()=='F' else 1,p.height_cm,p.gtp,p.hemoglobin,p.triglyceride,p.age,p.weight_kg,p.waist_cm,p.hdl,p.serum_creatinine,p.alt,p.fasting_blood_sugar,p.relaxation,p.ldl,p.systolic]],columns=features)

@app.get('/')
def root(): return {'message':'Bio Signal Analysis API','docs':'/docs'}
@app.get('/health')
def health(): return {'status':'ok'}
@app.post('/predict')
def predict(p:Patient):
    X=frame(p); xs=scaler.transform(X); pred=int(model.predict(xs)[0]); prob=float(model.predict_proba(xs)[0][1]) if hasattr(model,'predict_proba') else None
    return {'prediction':pred,'class':'smoker' if pred==1 else 'non-smoker','smoking_probability':prob}
