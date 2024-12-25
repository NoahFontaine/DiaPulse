import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import keras
from keras import layers, callbacks, models

class BloodGlucosePredictorB:
    def __init__(self, horizons):
        """
        Initialize the predictor with desired horizons.
        """
        self.horizons = horizons  # Prediction horizons
        self.scaler_x = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.model = None  # Placeholder for the LSTM model

    def _handle_missing_data(self, df: pd.DataFrame):
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
        df = self._handle_missing_data(df)

        # Convert 'Time' column to datetime if not already in datetime format
        if not np.issubdtype(df['Time'].dtype, np.datetime64):
            df['Time'] = pd.to_datetime(df['Time'])

        # Extract time-based features
        df['hour'] = df['Time'].dt.hour
        df['day_of_week'] = df['Time'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

        # Normalize time components
        df['hour_fraction'] = (df['hour'] + df['Time'].dt.minute / 60) / 24.0
        df['hour_sin'] = np.sin(2 * np.pi * df['hour_fraction'])
        df['hour_cos'] = np.cos(2 * np.pi * df['hour_fraction'])

        # Add lagging features for temporal context
        for lag in range(1, 4):
            df[f'CGM_lag_{lag}'] = df['CGM'].shift(lag)

        # Generate target columns for future predictions
        for h in self.horizons:
            df[f'BG_t+{h}'] = df['CGM'].shift(-int(h / 5))

        # Drop rows with NaN targets or lag features caused by shifting
        df.dropna(inplace=True)

        # Separate features and targets
        feature_cols = ['hour_sin', 'hour_cos', 'CGM', 'Insulin Activity', 'Food Activity'] + \
                       [f'CGM_lag_{lag}' for lag in range(1, 4)]
        X = df[feature_cols].values
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
            layers.LSTM(64, activation='relu', return_sequences=True, input_shape=input_shape),
            layers.Dropout(0.2),
            layers.LSTM(32, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(len(self.horizons))  # Output for each prediction horizon
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        self.model = model

    def train(self, X: np, y, validation_split=0.2, batch_size=32, epochs=200, model_save_path="best_model.keros"):
        """
        Train the LSTM model.
        """
        if self.model is None:
            raise ValueError("Model has not been built. Call build_model() first.")

        # Reshape input data for LSTM (samples, timesteps, features)
        X = X.reshape((X.shape[0], 1, X.shape[1]))

        early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
        model_checkpoint = callbacks.ModelCheckpoint(model_save_path, monitor='val_loss', save_best_only=True)

        history = self.model.fit(
            X, y,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, model_checkpoint],
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

    def load_model(self, model_path):
        """
        Load a saved model.
        """
        self.model = models.load_model(model_path)
