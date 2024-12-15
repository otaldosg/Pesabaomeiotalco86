import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from fastapi import FastAPI
import joblib
import json
from fastapi.responses import JSONResponse

app = FastAPI()

# Carregar dados do JSON (supondo que o arquivo JSON esteja no mesmo diretório ou em um diretório específico)
file_path = "dados.json"  # Substitua este caminho para o local do seu arquivo JSON

# Carregar os dados do arquivo JSON
with open(file_path, 'r') as file:
    data = json.load(file)

# Verifique se os dados estão no formato correto
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

# Endpoint para prever o fechamento do próximo dia
@app.get("/preverproximodia")
def prever_proximo_dia():
    # Prever o próximo fechamento
    latest_data = df.iloc[-1][['open', 'high', 'low', 'volume']].values.reshape(1, -1)
    predicted_close = model.predict(latest_data)

    return JSONResponse(content={"previsao": predicted_close[0]})

# Endpoint para escolher uma data específica para previsão
@app.get("/preverpordata")
def prever_por_data(data: str):
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
            return JSONResponse(content={"erro": "Data não encontrada no arquivo JSON."})

    except Exception as e:
        return JSONResponse(content={"erro": "Erro ao processar a data. Verifique o formato (YYYY-MM-DD) e tente novamente."})
