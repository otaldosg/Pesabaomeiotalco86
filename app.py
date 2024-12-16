import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from fastapi import FastAPI
import joblib
import json
from fastapi.responses import JSONResponse
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime
import requests

# Caminho para o arquivo JSON
file_path = "dados.json"

# URL da API Alpha Vantage
api_url = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=PETZ3.SAO&apikey=LVMKCDGMGNGO2C99"

app = FastAPI()

# Carregar modelo
model_save_path = "linear_regression_model.pkl"
model = None

def atualizar_dados():
    """Atualiza os dados do arquivo JSON a partir da API Alpha Vantage."""
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        data = response.json()

        # Salvar os dados no arquivo JSON
        with open(file_path, 'w') as file:
            json.dump(data, file)

        print(f"[{datetime.now()}] Dados atualizados com sucesso!")
    except Exception as e:
        print(f"[{datetime.now()}] Erro ao atualizar dados: {e}")

def carregar_e_treinar_modelo():
    """Carrega os dados do JSON, transforma-os em um DataFrame e treina o modelo."""
    global model
    try:
        # Carregar os dados do arquivo JSON
        with open(file_path, 'r') as file:
            data = json.load(file)

        # Verificar se os dados estão no formato correto
        if "Time Series (Daily)" in data:
            # Transformar os dados em um DataFrame
            df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient='index')
            df.columns = ['open', 'high', 'low', 'close', 'volume']
            df = df.astype(float)
            df.index = pd.to_datetime(df.index)  # Converte o índice para formato de data

            # Criar variáveis independentes (X) e dependentes (y)
            X = df[['open', 'high', 'low', 'volume']]
            y = df['close']

            # Dividir os dados em conjuntos de treinamento e teste
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Criar e treinar o modelo
            model = LinearRegression()
            model.fit(X_train, y_train)

            # Salvar o modelo
            joblib.dump(model, model_save_path)
            print(f"[{datetime.now()}] Modelo treinado e salvo com sucesso!")
    except Exception as e:
        print(f"[{datetime.now()}] Erro ao carregar e treinar o modelo: {e}")

# Inicializar o agendador de tarefas
scheduler = BackgroundScheduler(timezone="America/Sao_Paulo")
scheduler.add_job(atualizar_dados, "cron", hour=6, minute=0)  # Atualiza às 6h da manhã
scheduler.start()

# Atualizar os dados e treinar o modelo ao iniciar
atualizar_dados()
carregar_e_treinar_modelo()

@app.get("/preverproximodia")
def prever_proximo_dia():
    try:
        # Carregar os dados
        with open(file_path, 'r') as file:
            data = json.load(file)

        if "Time Series (Daily)" in data:
            df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient='index')
            df.columns = ['open', 'high', 'low', 'close', 'volume']
            df = df.astype(float)
            df.index = pd.to_datetime(df.index)

            # Prever o próximo fechamento
            latest_data = df.iloc[0][['open', 'high', 'low', 'volume']].values.reshape(1, -1)
            predicted_close = model.predict(latest_data)

            return JSONResponse(content={"previsao": predicted_close[0]})
        else:
            return JSONResponse(content={"erro": "Formato inválido no arquivo JSON."})
    except Exception as e:
        return JSONResponse(content={"erro": f"Erro ao realizar a previsão: {e}"})
