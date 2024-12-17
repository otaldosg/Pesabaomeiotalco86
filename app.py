import requests
import pandas as pd
import json
from datetime import datetime
import schedule
import time
import pytz
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import joblib

app = FastAPI()

# Configurações
ALPHA_VANTAGE_API_URL = "https://www.alphavantage.co/query"
API_KEY = "LVMKCDGMGNGO2C99"
SYMBOL = "PETZ3.SAO"
FILE_PATH = "dados.json"
MODEL_SAVE_PATH = "linear_regression_model.pkl"

# Função para atualizar os dados do Alpha Vantage
def atualizar_dados():
    try:
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": SYMBOL,
            "apikey": API_KEY
        }
        print(f"Atualizando dados às {datetime.now(pytz.timezone('America/Sao_Paulo'))}...")
        
        response = requests.get(ALPHA_VANTAGE_API_URL, params=params)
        if response.status_code == 200:
            data = response.json()
            if "Time Series (Daily)" in data:
                # Salvar os dados no arquivo JSON
                with open(FILE_PATH, "w") as f:
                    json.dump(data, f, indent=4)
                print("Dados atualizados com sucesso!")
            else:
                print("Erro: Resposta da API não contém os dados esperados.")
        else:
            print(f"Erro ao acessar a API: {response.status_code}")
    except Exception as e:
        print(f"Erro durante a atualização dos dados: {e}")

# Função para treinar o modelo e carregar os dados
def carregar_e_treinar_modelo():
    global df, model
    with open(FILE_PATH, 'r') as file:
        data = json.load(file)

    if "Time Series (Daily)" in data:
        df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient='index')
        df.columns = ['open', 'high', 'low', 'close', 'volume']
        df = df.astype(float)
        df.index = pd.to_datetime(df.index)

        X = df[['open', 'high', 'low', 'volume']]
        y = df['close']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f'Mean Squared Error: {mse}')

        joblib.dump(model, MODEL_SAVE_PATH)
        print(f"Modelo treinado e salvo em: {MODEL_SAVE_PATH}")

# Agendamento para atualizar os dados diariamente às 6h (horário de Brasília)
def agendar_atualizacao():
    schedule.every().day.at("06:00").do(atualizar_dados)

    while True:
        schedule.run_pending()
        time.sleep(60)  # Checa a cada 60 segundos

# Inicializar o modelo ao iniciar
atualizar_dados()  # Primeira atualização imediata
carregar_e_treinar_modelo()

# Rota para prever o fechamento do próximo dia
@app.get("/preverproximodia")
def prever_proximo_dia():
    latest_data = df.iloc[0][['open', 'high', 'low', 'volume']].values.reshape(1, -1)
    predicted_close = model.predict(latest_data)
    return JSONResponse(content={"previsao": predicted_close[0]})

# Rota para prever uma data específica
@app.get("/preverpordata")
def prever_por_data(data: str):
    try:
        selected_date = pd.to_datetime(data)

        if selected_date in df.index:
            filtered_df = df[df.index < selected_date]
            if len(filtered_df) > 1:
                X_filtered = filtered_df[['open', 'high', 'low', 'volume']]
                y_filtered = filtered_df['close']

                model.fit(X_filtered, y_filtered)
                selected_day_data = df.loc[selected_date][['open', 'high', 'low', 'volume']].values.reshape(1, -1)
                predicted_close = model.predict(selected_day_data)

                actual_close = df.loc[selected_date]['close']
                return JSONResponse(content={
                    "previsao": predicted_close[0],
                    "valor_real": actual_close
                })
            else:
                return JSONResponse(content={"erro": "Dados insuficientes para realizar a previsão."})
        else:
            return JSONResponse(content={"erro": "Data não encontrada no arquivo JSON."})
    except Exception as e:
        return JSONResponse(content={"erro": "Erro ao processar a data."})

# Agendamento rodando em segundo plano
import threading
t = threading.Thread(target=agendar_atualizacao)
t.start()
