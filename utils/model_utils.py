import numpy as np

def create_sequences(data, window):
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i - window:i])
        y.append(data[i])
    return np.array(X), np.array(y)

def forecast_future_days(model, data_scaled, n_days, scaler):
    input_seq = data_scaled[-60:]
    future_preds = []
    for _ in range(n_days):
        pred = model.predict(input_seq.reshape(1, 60, 1), verbose=0)[0][0]
        future_preds.append(pred)
        input_seq = np.append(input_seq[1:], [[pred]], axis=0)
    return scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))

