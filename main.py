# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd

# Initialize FastAPI app
app = FastAPI()


# Load the trained model and LabelEncoder
try:
    model = joblib.load('random_forest_gad7_model.joblib')
    label_encoder = joblib.load('label_encoder.joblib')
except FileNotFoundError:
    raise RuntimeError("Model or LabelEncoder file not found. Please ensure 'random_forest_gad7_model.joblib' and 'label_encoder.joblib' are in the same directory.")

# Mapping from string answers to numerical scores
ANSWER_MAPPING = {
    'Tidak Pernah': 0,
    'Beberapa Hari': 1,
    'Lebih dari Separuh Waktu yang ditentukan': 2,
    'Hampir Setiap Hari': 3
}

# Define the input data model for the API
class GAD7Input(BaseModel):
    # Questions from the GAD-7, accepting string answers
    
    merasa_gugup_cemas_atau_gelisah: str = Field(
        ..., 
        description="Merasa gugup, cemas, atau gelisah (Pilih salah satu: 'Tidak Pernah', 'Beberapa Hari', 'Lebih dari Separuh Waktu yang ditentukan', 'Hampir Setiap Hari')"
    )
    
    tidak_dapat_menghentikan_kekhawatiran: str = Field(
        ..., 
        description="Tidak dapat menghentikan kekhawatiran (Pilih salah satu: 'Tidak Pernah', 'Beberapa Hari', 'Lebih dari Separuh Waktu yang ditentukan', 'Hampir Setiap Hari')"
    )
    
    banyak_mengkhawatirkan_berbagai_hal: str = Field(
        ..., 
        description="Banyak mengkhawatirkan berbagai hal (Pilih salah satu: 'Tidak Pernah', 'Beberapa Hari', 'Lebih dari Separuh Waktu yang ditentukan', 'Hampir Setiap Hari')"
    )
    
    sulit_merasa_santai: str = Field(
        ..., 
        description="Sulit merasa santai (Pilih salah satu: 'Tidak Pernah', 'Beberapa Hari', 'Lebih dari Separuh Waktu yang ditentukan', 'Hampir Setiap Hari')"
    )
    
    sangat_gelisah_sehingga_sulit_untuk_diam: str = Field(
        ..., 
        description="Sangat gelisah sehingga sulit untuk diam (Pilih salah satu: 'Tidak Pernah', 'Beberapa Hari', 'Lebih dari Separuh Waktu yang ditentukan', 'Hampir Setiap Hari')"
    )
    
    mudah_tersinggung_dan_mudah_marah: str = Field(
        ..., 
        description="Mudah tersinggung dan mudah marah (Pilih salah satu: 'Tidak Pernah', 'Beberapa Hari', 'Lebih dari Separuh Waktu yang ditentukan', 'Hampir Setiap Hari')"
    )
    
    merasa_takut_seolah_olah_sesuatu_buruk_akan_terjadi: str = Field(
        ..., 
        description="Merasa takut seolah-olah sesuatu buruk akan terjadi (Pilih salah satu: 'Tidak Pernah', 'Beberapa Hari', 'Lebih dari Separuh Waktu yang ditentukan', 'Hampir Setiap Hari')"
    )






# Define the response data model
class PredictionOutput(BaseModel):
    total_score: int
    anxiety_level: str
    anxiety_label_encoded: int

@app.get("/")
async def read_root():
    return {"message": "Welcome to the GAD-7 Anxiety Level Prediction API"}

@app.post("/predict", response_model=PredictionOutput)
async def predict_anxiety(input_data: GAD7Input):
    # Original column names from your notebook's df_clean
    original_column_names = [
        'Merasa gugup, cemas, atau gelisah',
        'Tidak dapat menghentikan kekhawatiran',
        'Banyak mengkhawatirkan berbagai hal',
        'Sulit merasa santai',
        'Sangat gelisah sehingga sulit untuk diam',
        'Mudah tersinggung dan mudah marah',
        'Merasa takut seolah-olah sesuatu buruk akan terjadi'
    ]
    
    # Convert string answers to numerical scores using ANSWER_MAPPING
    processed_input = {}
    
    try:
        processed_input[original_column_names[0]] = ANSWER_MAPPING[input_data.merasa_gugup_cemas_atau_gelisah]
        processed_input[original_column_names[1]] = ANSWER_MAPPING[input_data.tidak_dapat_menghentikan_kekhawatiran]
        processed_input[original_column_names[2]] = ANSWER_MAPPING[input_data.banyak_mengkhawatirkan_berbagai_hal]
        processed_input[original_column_names[3]] = ANSWER_MAPPING[input_data.sulit_merasa_santai]
        processed_input[original_column_names[4]] = ANSWER_MAPPING[input_data.sangat_gelisah_sehingga_sulit_untuk_diam]
        processed_input[original_column_names[5]] = ANSWER_MAPPING[input_data.mudah_tersinggung_dan_mudah_marah]
        processed_input[original_column_names[6]] = ANSWER_MAPPING[input_data.merasa_takut_seolah_olah_sesuatu_buruk_akan_terjadi]
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Invalid answer provided: {e}. Please use one of the allowed string values.")

    input_df = pd.DataFrame([processed_input])

    # Predict the anxiety level (encoded label)
    prediction_encoded = model.predict(input_df)[0]

    # Decode the numerical prediction back to the original anxiety level string
    anxiety_level = label_encoder.inverse_transform([prediction_encoded])[0]

    # Calculate total score
    total_score = input_df.sum(axis=1).iloc[0]

    return {
        "total_score": total_score,
        "anxiety_level": anxiety_level,
        "anxiety_label_encoded": prediction_encoded
        
    }


