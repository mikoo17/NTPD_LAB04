from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import os
from sklearn.linear_model import LinearRegression

# Inicjalizacja aplikacji
app = FastAPI()

# Przykładowy model ML
model = LinearRegression()
X_train = np.array([[1], [2], [3], [4], [5]])
y_train = np.array([2, 4, 6, 8, 10])
model.fit(X_train, y_train)
api_key = os.getenv("API_KEY", "")
# Definicja schematu danych wejściowych
class InputData(BaseModel):
    value: float

# Endpoint główny
@app.get("/")
def read_root():
    return {"message": "Witaj w API!"}

# Endpoint predykcji
@app.post("/predict")
def predict(data: InputData):
    try:
        prediction = model.predict(np.array([[data.value]]))[0]
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Endpoint informacji o modelu
@app.get("/info")
def model_info():
    return {"model": "LinearRegression", "features": 1}

# Endpoint sprawdzania statusu serwera
@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "api_key": api_key if api_key else "Not Set"  # Pokazuje, czy klucz API jest ustawiony
    }
