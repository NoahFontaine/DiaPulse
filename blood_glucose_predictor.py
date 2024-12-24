import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import keras
from keras import layers, callbacks

class BloodGlucosePredictor:
    def __init__(self, horizons):
        """
        Initialize the predictor with desired horizons.
        """
        self.horizons = horizons  # Prediction horizons
        self.scaler_x = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.model = None  # Placeholder for the LSTM model

    def handle_missing_data(self, df: pd.DataFrame):
        """
        Handle missing data by interpolating the gaps.
        """
        df.interpolate(method='linear', inplace=True)  # Linear interpolation for missing values
        return df

    def preprocess_data(self, df):
        """
        Preprocess the dataset: handle missing data, generate targets, and scale features/targets.
        """
        # Handle missing data
        df = self.handle_missing_data(df)

        # Generate target columns for future predictions
        for h in self.horizons:
            df[f'BG_t+{h}'] = df['CGM'].shift(-h)

        # Drop rows with NaN targets caused by shifting
        df.dropna(inplace=True)

        # Separate features and targets
        X = df[['CGM', 'Insulin', 'Food']].values
        y = df[[f'BG_t+{h}' for h in self.horizons]].values

        # Scale features and targets
        X_scaled = self.scaler_x.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y)

        return X_scaled, y_scaled

    def build_model(self, input_shape):
        """
        Build the LSTM model.
        """
        model = keras.Sequential([
            layers.LSTM(64, activation='relu', input_shape=input_shape),
            layers.Dense(len(self.horizons))  # Output for each prediction horizon
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        self.model = model

    def train(self, X: np, y, validation_split=0.2, batch_size=32, epochs=100):
        """
        Train the LSTM model.
        """
        if self.model is None:
            raise ValueError("Model has not been built. Call build_model() first.")

        # Reshape input data for LSTM (samples, timesteps, features)
        X = X.reshape((X.shape[0], 1, X.shape[1]))

        early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        history = self.model.fit(
            X, y,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=1
        )
        return history

    def evaluate(self, X: np, y):
        """
        Evaluate the model on test data.
        """
        if self.model is None:
            raise ValueError("Model has not been built. Call build_model() first.")

        # Reshape input data for LSTM (samples, timesteps, features)
        X = X.reshape((X.shape[0], 1, X.shape[1]))

        loss, mae = self.model.evaluate(X, y, verbose=0)
        return loss, mae

    def predict(self, X: np):
        """
        Make predictions with the trained model.
        """
        if self.model is None:
            raise ValueError("Model has not been built. Call build_model() first.")

        # Reshape input data for LSTM (samples, timesteps, features)
        X = X.reshape((X.shape[0], 1, X.shape[1]))

        y_pred_scaled = self.model.predict(X)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
        return y_pred
