# noshow_iq/api.py
import os
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from . import model, preprocess, db
 
app = FastAPI(title='NoShowIQ', version='1.0')
 
class Appointment(BaseModel):
    Gender: str
    Age: int = Field(ge=0, le=110)
    scheduled_day: str   # ISO datetime
    appointment_day: str
    Scholarship: int = 0
    hypertension: int = 0
    Diabetes: int = 0
    Alcoholism: int = 0
    handicap: int = 0
    sms_received: int = 0
 
@app.get('/health')
def health():
    return {'status': 'ok', 'time': datetime.now(timezone.utc).isoformat()}
 
@app.post('/predict')
def predict(appt: Appointment):
    try:
        import pandas as pd
        raw = appt.model_dump()
        df = pd.DataFrame([raw])
        df = preprocess.engineer(preprocess.clean(df))
        # match feature order from training
        result = model.predict(df.iloc[0].to_dict())
    except Exception as e:
        raise HTTPException(500, str(e))
    db.log_prediction(raw_input=raw, cleaned=df.iloc[0].to_dict(), result=result)
    return result
 
@app.get('/history')
def history():
    return db.last_n_predictions(20)
 
@app.get('/stats')
def stats():
    return db.aggregate_stats()