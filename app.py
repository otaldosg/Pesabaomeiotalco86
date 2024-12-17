import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from fastapi import FastAPI
import joblib
import json
import requests
import threading
from datetime import datetime, timedelta
from fastapi.responses import JSONResponse

app = FastAPI()

# Função para buscar dados da API Alpha Vantage
def buscar_dados():
    api_key = "LVMKCDGMGNGO2C99"  # Sua chave de API
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": "PETZ3.SAO",
        "apikey": api_key
    }
    response = requests.get(url, params=params)
    data = response.json()

    return data

# Função para atualizar o arquivo dados.json
def atualizar_dados():
    data = buscar_dados()

    # Salvar os dados no arquivo JSON
    with open("dados.json", "w") as file:
        json.dump(data, file)
    
    print(f"Dados atualizados em: {datetime.now()}")

    # Após a atualização, refazer o modelo com os novos dados
    processar_dados()

# Função para processar os dados e treinar o modelo
def processar_dados():
    # Carregar os dados do arquivo JSON
    with open("dados.json", "r") as file:
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

        # Fazer previsões no conjunto de teste para avaliação
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f'Mean Squared Error: {mse}')

        # Salvar o modelo
        model_save_path = "linear_regression_model.pkl"
        joblib.dump(model, model_save_path)
        print(f'Modelo salvo em: {model_save_path}')

# Função que agenda a atualização a cada 4 horas
def agendar_atualizacao():
    while True:
        now = datetime.now()
        
        # Definir os horários de atualização (00:00, 04:00, 08:00, 12:00, 16:00, 20:00, 22:00)
        update_hours = [0, 4, 8, 12, 16, 20, 22]

        # Encontrar o próximo horário de atualização
        next_update_hour = min([hour for hour in update_hours if hour > now.hour], default=0)

        # Se o próximo horário for 00:00 (início de um novo ciclo), ajustar para o próximo dia
        if next_update_hour == 0:
            next_update = now.replace(hour=next_update_hour, minute=0, second=0, microsecond=0) + timedelta(days=1)
        else:
            next_update = now.replace(hour=next_update_hour, minute=0, second=0, microsecond=0)

        # Calcular o tempo restante até o próximo horário de atualização
        time_to_wait = (next_update - now).total_seconds()
        print(f"A próxima atualização será às {next_update.strftime('%H:%M:%S')}")

        # Esperar até o próximo horário de atualização
        threading.Timer(time_to_wait, atualizar_dados).start()
        break

# Endpoint para prever o fechamento do próximo dia
@app.get("/preverproximodia")
def prever_proximo_dia():
    # Carregar o modelo treinado
    model = joblib.load("linear_regression_model.pkl")

    # Carregar os dados mais recentes
    with open("dados.json", "r") as file:
        data = json.load(file)

    df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient='index')
    df.columns = ['open', 'high', 'low', 'close', 'volume']
    df = df.astype(float)
    df.index = pd.to_datetime(df.index)  # Converte o índice para formato de data

    # Prever o próximo fechamento
    latest_data = df.iloc[0][['open', 'high', 'low', 'volume']].values.reshape(1, -1)
    predicted_close = model.predict(latest_data)

    return JSONResponse(content={"previsao": predicted_close[0]})

# Endpoint para escolher uma data específica para previsão
@app.get("/preverpordata")
def prever_por_data(data: str):
    # Carregar o modelo treinado
    model = joblib.load("linear_regression_model.pkl")

    try:
        selected_date = pd.to_datetime(data)

        # Carregar os dados mais recentes
        with open("dados.json", "r") as file:
            data = json.load(file)

        df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient='index')
        df.columns = ['open', 'high', 'low', 'close', 'volume']
        df = df.astype(float)
        df.index = pd.to_datetime(df.index)  # Converte o índice para formato de data

        if selected_date in df.index:
            # Filtrar os dados até a data selecionada
            filtered_df = df[df.index < selected_date]

            # Verificar se há dados suficientes
            if len(filtered_df) > 1:
                X_filtered = filtered_df[['open', 'high', 'low', 'volume']]
                y_filtered = filtered_df['close']

                # Treinar o modelo com os dados até a data selecionada
                model.fit(X_filtered, y_filtered)

                # Prever o fechamento para a data escolhida
                selected_day_data = df.loc[selected_date][['open', 'high', 'low', 'volume']].values.reshape(1, -1)
                predicted_close = model.predict(selected_day_data)

                # Valor real
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
        return JSONResponse(content={"erro": "Erro ao processar a data. Verifique o formato (YYYY-MM-DD) e tente novamente."})

# Iniciar o agendamento de atualizações
agendar_atualizacao()
