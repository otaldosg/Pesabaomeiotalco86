import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from fastapi import FastAPI
import joblib
import requests
import sqlite3
from FastAPI import FileResponse
from fastapi.responses import JSONResponse
import json
import os

app = FastAPI()

# URL da API da Alpha Vantage
ALPHA_VANTAGE_API_URL = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=PETZ3.SAO&apikey=LVMKCDGMGNGO2C99"

# Função para buscar dados da API e atualizar o banco de dados
def atualizar_dados():
    response = requests.get(ALPHA_VANTAGE_API_URL)
    if response.status_code == 200:
        data = response.json()
        
        # Conectar ao banco de dados SQLite
        conn = sqlite3.connect("dados.db")
        cursor = conn.cursor()

        # Criar a tabela, se não existir
        cursor.execute('''CREATE TABLE IF NOT EXISTS time_series (
                            date TEXT PRIMARY KEY,
                            open REAL,
                            high REAL,
                            low REAL,
                            close REAL,
                            volume REAL)''')

        # Limpar a tabela antes de inserir os dados novos
        cursor.execute("DELETE FROM time_series")

        # Inserir os dados da API no banco de dados
        for date, values in data["Time Series (Daily)"].items():
            cursor.execute('''INSERT OR REPLACE INTO time_series (date, open, high, low, close, volume)
                            VALUES (?, ?, ?, ?, ?, ?)''', 
                            (date, 
                             values['1. open'], 
                             values['2. high'], 
                             values['3. low'], 
                             values['4. close'], 
                             values['5. volume']))

        conn.commit()
        conn.close()
    else:
        print("Erro ao buscar dados da API.")

# Carregar dados do banco de dados
def carregar_dados():
    conn = sqlite3.connect("dados.db")
    df = pd.read_sql_query("SELECT * FROM time_series", conn)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    conn.close()
    return df

# Atualizar os dados ao iniciar
atualizar_dados()

# Carregar os dados atualizados
df = carregar_dados()

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

# Endpoint para prever o fechamento do próximo dia
@app.get("/preverproximodia")
def prever_proximo_dia():
    # Atualizar dados sempre que o endpoint for chamado
    atualizar_dados()

    # Carregar os dados atualizados
    df = carregar_dados()

    # Prever o próximo fechamento
    latest_data = df.iloc[0][['open', 'high', 'low', 'volume']].values.reshape(1, -1)
    predicted_close = model.predict(latest_data)

    return JSONResponse(content={"previsao": predicted_close[0]})

@app.get("/dados")
def baixar_dados():
    try:
        # Verificar se o arquivo do banco de dados existe
        if not os.path.exists("dados.db"):
            return JSONResponse(content={"erro": "Banco de dados não encontrado."}, status_code=404)

        # Retornar o arquivo dados.db para o usuário
        return FileResponse("dados.db", media_type='application/octet-stream', filename="dados.db")
        
    except Exception as e:
        # Log do erro para debug
        print(f"Erro ao acessar os dados do banco: {e}")
        return JSONResponse(content={"erro": f"Erro ao acessar os dados do banco: {e}"}, status_code=500)
        
# Endpoint para escolher uma data específica para previsão
@app.get("/preverpordata")
def prever_por_data(data: str):
    # Atualizar dados sempre que o endpoint for chamado
    atualizar_dados()

    # Carregar os dados atualizados
    df = carregar_dados()

    try:
        selected_date = pd.to_datetime(data)

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
            return JSONResponse(content={"erro": "Data não encontrada no banco de dados."})

    except Exception as e:
        return JSONResponse(content={"erro": "Erro ao processar a data. Verifique o formato (YYYY-MM-DD) e tente novamente."})
