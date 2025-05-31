###  Cara Menjalankan

**Install Dependencies**

```python
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```
**Install dependencies:**

```python
pip install fastapi uvicorn scikit-learn pandas joblib pydantic
```

Jalankan API menggunakan uvicorn:

```python
uvicorn main:app --reload
```

**FastAPI akan tersedia di:**

http://127.0.0.1:8000


### Endpoint


POST /predict

Request Body:

{
  "merasa_gugup_cemas_atau_gelisah": "Beberapa Hari",
  "tidak_dapat_menghentikan_kekhawatiran": "Beberapa Hari",
  "banyak_mengkhawatirkan_berbagai_hal": "Beberapa Hari",
  "sulit_merasa_santai": "Beberapa Hari",
  "sangat_gelisah_sehingga_sulit_untuk_diam": "Beberapa Hari",
  "mudah_tersinggung_dan_mudah_marah": "Beberapa Hari",
  "merasa_takut_seolah_olah_sesuatu_buruk_akan_terjadi": "Beberapa Hari"
}


Response:

{
  "total_score": 7,
  "anxiety_level": "Sedang",
  "anxiety_label_encoded": 1
}

