from sklearn.ensemble import IsolationForest
from keras.models import Model, Sequential
from keras.layers import LSTM, Dense, RepeatVector, TimeDistributed
import numpy as np

def train_isolation_forest(data):
    """
    Train an Isolation Forest model for anomaly detection.
    Args:
        data (pd.DataFrame or np.ndarray): Preprocessed feature data.
    Returns:
        IsolationForest: Trained Isolation Forest model.
    """
    model = IsolationForest(contamination='auto', random_state=42)
    model.fit(data)
    return model

def create_lstm_autoencoder(timesteps, n_features):
    """
    Build an LSTM-based autoencoder model for time series anomaly detection.
    Args:
        timesteps (int): Number of time steps in each input sample.
        n_features (int): Number of features in each time step.
    Returns:
        keras.Model: Compiled LSTM autoencoder model.
    """
    model = Sequential()
    model.add(LSTM(128, activation='relu', input_shape=(timesteps, n_features), return_sequences=True))
    model.add(LSTM(64, activation='relu', return_sequences=False))
    model.add(RepeatVector(timesteps))
    model.add(LSTM(64, activation='relu', return_sequences=True))
    model.add(LSTM(128, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(n_features)))
    model.compile(optimizer='adam', loss='mse')
    return model

def train_autoencoder(data, timesteps=10):
    """
    Train an LSTM autoencoder for time series anomaly detection.
    Args:
        data (pd.DataFrame): Preprocessed feature data.
        timesteps (int): Number of time steps in each input sample.
    Returns:
        keras.Model: Trained LSTM autoencoder model.
    """
    # Clear Keras backend session to avoid state issues in Streamlit
    from keras import backend as K
    K.clear_session()
    # Reshape data for LSTM
    n_samples = data.shape[0] - timesteps
    n_features = data.shape[1]
    X = np.zeros((n_samples, timesteps, n_features))
    for i in range(n_samples):
        X[i] = data.iloc[i:i+timesteps].values

    model = create_lstm_autoencoder(timesteps, n_features)
    model.fit(X, X, epochs=20, batch_size=32, validation_split=0.1, verbose=0)
    return model

def predict_anomalies(model, data, contamination=0.1, threshold=0.1):
    """
    Predict anomalies using a trained model (Isolation Forest or LSTM Autoencoder).
    Args:
        model: Trained model (IsolationForest or keras.Model).
        data (pd.DataFrame): Preprocessed feature data.
        contamination (float): Used for Isolation Forest.
        threshold (float): MSE threshold for autoencoder anomaly detection.
    Returns:
        np.ndarray: Array of anomaly labels (1 for normal, -1 for anomaly).
    """
    if isinstance(model, IsolationForest):
        return model.predict(data)
    else: # Autoencoder
        timesteps = model.input_shape[1]
        n_samples = data.shape[0] - timesteps
        n_features = data.shape[1]
        X = np.zeros((n_samples, timesteps, n_features))
        for i in range(n_samples):
            X[i] = data.iloc[i:i+timesteps].values
        
        reconstructions = model.predict(X)
        mse = np.mean(np.power(X - reconstructions, 2), axis=(1, 2))
        
        anomalies = np.full(data.shape[0], 1) # 1 for normal
        # The first `timesteps` samples cannot be evaluated
        anomalies[timesteps:][mse > threshold] = -1 # -1 for anomaly
        return anomalies
