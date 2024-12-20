import numpy as np
import pandas as pd
import json
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI()

# Caminho para o arquivo JSON
file_path = "dados.json"

# Carregar os dados do JSON
with open(file_path, 'r') as file:
    data = json.load(file)

# Verificar e transformar os dados
if "Time Series (Daily)" in data:
    df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient="index")
    df.columns = ['open', 'high', 'low', 'close', 'volume']
    df = df.astype(float)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()  # Ordenar por data

    # Normalizar os dados
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[['close']])

    # Criar sequências de dados para LSTM
    def create_sequences(data, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length, 0])
            y.append(data[i + seq_length, 0])
        return np.array(X), np.array(y)

    seq_length = 30  # Número de dias usados para previsão
    X, y = create_sequences(scaled_data, seq_length)

    # Dividir os dados em treino e teste
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Redimensionar para o formato LSTM
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # Criar o modelo LSTM
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(seq_length, 1)))
    model.add(LSTM(50))
    model.add(Dense(1))

    # Compilar o modelo
    model.compile(optimizer='adam', loss='mse')

    # Treinar o modelo
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

    # Avaliar o modelo
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error (LSTM): {mse}")

# Endpoint para prever o fechamento do próximo dia
@app.get("/preverproximodia")
def prever_proximo_dia():
    # Obter os últimos dados para previsão
    latest_data = df.iloc[-SEQUENCE_LENGTH:][['open', 'high', 'low', 'volume']].values
    latest_data_scaled = scaler.transform(latest_data)
    latest_data_reshaped = latest_data_scaled.reshape(1, SEQUENCE_LENGTH, latest_data_scaled.shape[1])

    # Fazer a previsão
    predicted_close = model.predict(latest_data_reshaped)
    predicted_close = predicted_close[0][0]  # Obter o valor da previsão
    predicted_close = float(predicted_close)  # Converter para float padrão

    return JSONResponse(content={"previsao": predicted_close})
    
# Endpoint para obter MSE
@app.get("/mse")
def get_mse():
    return JSONResponse(content={"mean_squared_error": mse})
