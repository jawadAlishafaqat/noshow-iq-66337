# noshow_iq/preprocess.py
import pandas as pd
import numpy as np
 
RENAME_MAP = {
    'No-show': 'no_show',
    'Hipertension': 'hypertension',
    'Handcap': 'handicap',
    'SMS_received': 'sms_received',
    'ScheduledDay': 'scheduled_day',
    'AppointmentDay': 'appointment_day',
}
 
def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.rename(columns=RENAME_MAP)
 
def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['scheduled_day'] = pd.to_datetime(df['scheduled_day'], utc=True)
    df['appointment_day'] = pd.to_datetime(df['appointment_day'], utc=True)
    df = df[df['Age'] >= 0]
    df = df[df['Age'] <= 110]
    df['no_show'] = (df['no_show'] == 'Yes').astype(int)  # 1 = no-show
    return df
 
def engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['days_in_advance'] = (
        df['appointment_day'].dt.normalize() - df['scheduled_day'].dt.normalize()
    ).dt.days
    df.loc[df['days_in_advance'] < 0, 'days_in_advance'] = 0  # fix negative gaps
    df['appointment_dow'] = df['appointment_day'].dt.dayofweek
    df['age_group'] = pd.cut(
        df['Age'], bins=[-1, 12, 18, 35, 60, 120],
        labels=['child', 'teen', 'young_adult', 'adult', 'senior']
    )
    return df
 
FEATURE_COLS = [
    'Age', 'days_in_advance', 'appointment_dow', 'sms_received',
    'Scholarship', 'hypertension', 'Diabetes', 'Alcoholism', 'handicap',
    # one-hot encode age_group and Gender in the pipeline
]
 
def prepare(df: pd.DataFrame):
    """Full pipeline. Returns (X, y) ready for model.train()."""
    df = clean(df)
    df = engineer(df)
    df = pd.get_dummies(df, columns=['Gender', 'age_group'], drop_first=True)
    feature_cols = [c for c in df.columns if c not in
                    ('no_show', 'PatientId', 'AppointmentID',
                     'scheduled_day', 'appointment_day', 'Neighbourhood')]
    return df[feature_cols], df['no_show']