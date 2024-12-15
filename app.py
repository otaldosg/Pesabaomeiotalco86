from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd
import joblib
import json
import os

# Inicializar a aplicação FastAPI
app = FastAPI()

# Definir caminhos para os arquivos
DATA_FILE = "dados.json"
MODEL_FILE = "linear_regression_model.pkl"

# Inicializar variáveis globais
df = None
model = None

# Função para carregar dados e modelo
def load_data_and_model():
    global df, model

    # Carregar os dados
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            data = json.load(f)

        if "Time Series (Daily)" in data:
            df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient="index")
            df.columns = ["open", "high", "low", "close", "volume"]
            df = df.astype(float)
            df.index = pd.to_datetime(df.index)
        else:
            raise ValueError("Formato inválido no arquivo JSON.")
    else:
        raise FileNotFoundError("Arquivo 'dados.json' não encontrado.")

    # Carregar o modelo ou treinar um novo
    if os.path.exists(MODEL_FILE):
        model = joblib.load(MODEL_FILE)
    else:
        X = df[["open", "high", "low", "volume"]]
        y = df["close"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        joblib.dump(model, MODEL_FILE)

# Inicializar dados e modelo ao iniciar o servidor
load_data_and_model()

# Modelo de entrada para previsões personalizadas
class PredictionRequest(BaseModel):
    open: float
    high: float
    low: float
    volume: float

# Rota inicial
@app.get("/")
def home():
    return {"message": "Bem-vindo à API de Previsão de Fechamento!"}

# Rota para prever o fechamento do próximo dia
@app.get("/previsao/")
def previsao(tipo: str = "proximodia", data: str = None):
    if tipo == "proximodia":
        latest_data = df.iloc[-1][["open", "high", "low", "volume"]].values.reshape(1, -1)
        predicted_close = model.predict(latest_data)
        return {"next_close_prediction": predicted_close[0]}

    elif tipo == "dataprevisao" and data:
        try:
            selected_date = pd.to_datetime(data)
            if selected_date in df.index:
                filtered_df = df[df.index < selected_date]
                if len(filtered_df) > 1:
                    X_filtered = filtered_df[["open", "high", "low", "volume"]]
                    y_filtered = filtered_df["close"]
                    model.fit(X_filtered, y_filtered)
                    selected_day_data = df.loc[selected_date][["open", "high", "low", "volume"]].values.reshape(1, -1)
                    predicted_close = model.predict(selected_day_data)
                    actual_close = df.loc[selected_date]["close"]
                    return {
                        "date": data,
                        "predicted_close": predicted_close[0],
                        "actual_close": actual_close,
                    }
                else:
                    raise HTTPException(status_code=400, detail="Dados insuficientes para previsão.")
            else:
                raise HTTPException(status_code=404, detail="Data não encontrada.")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Erro ao processar a data: {str(e)}")
    else:
        raise HTTPException(status_code=400, detail="Parâmetro 'tipo' inválido ou faltando.")

# Rota para previsões personalizadas
@app.post("/predict-custom/")
def predict_custom_data(data: PredictionRequest):
    features = [[data.open, data.high, data.low, data.volume]]
    predicted_close = model.predict(features)
    return {"custom_prediction": predicted_close[0]}
