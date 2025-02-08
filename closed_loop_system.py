import numpy as np
import pandas as pd
from models.model_F import BloodGlucosePredictorF

class ClosedLoopSystem:
    def __init__(self, predictor, target_bg=100):
        self.predictor = predictor
        self.target_bg = target_bg
        self.current_bg = None
        self.insulin_dose = 0

    def control_algorithm(self, predicted_bg):
        """
        Simple proportional control algorithm to adjust insulin dose.
        """
        error = self.target_bg - predicted_bg
        k_p = 0.1  # Proportional gain, adjust as needed
        insulin_adjustment = k_p * error
        self.insulin_dose += insulin_adjustment
        return self.insulin_dose

    def simulate(self, initial_data, steps=100):
        """
        Simulate the closed-loop system.
        """
        data = initial_data.copy()
        for step in range(steps):
            # Preprocess data
            X, _ = self.predictor.preprocess_data(data)

            # Predict future blood glucose levels
            predicted_bg = self.predictor.predict(X[-1].reshape(1, X.shape[1], X.shape[2]))[0]

            # Adjust insulin dose based on prediction
            self.insulin_dose = self.control_algorithm(predicted_bg)

            # Update current blood glucose level (for simulation purposes)
            self.current_bg = predicted_bg + np.random.normal(0, 5)  # Add some noise for realism

            # Append new data point to the dataset
            new_data_point = {
                'Time': data['Time'].iloc[-1] + pd.Timedelta(minutes=5),
                'CGM': self.current_bg,
                'Insulin Activity': self.insulin_dose,
                'Food Activity': 0,  # Assuming no food intake for simplicity
                'CGM Gradient': self.current_bg - data['CGM'].iloc[-1]
            }
            data = data.append(new_data_point, ignore_index=True)

            print(f"Step {step + 1}: Predicted BG = {predicted_bg:.2f}, Insulin Dose = {self.insulin_dose:.2f}")

# Example usage
horizons = [5, 10, 15, 20, 25, 30, 60, 90, 120]
predictor = BloodGlucosePredictorF(horizons)
predictor.build_model(input_shape=(10, len(predictor.preprocess_data(pd.DataFrame())[0][0])))

# Load initial data (replace with your actual data)
initial_data = pd.DataFrame({
    'Time': pd.date_range(start='2025-02-08', periods=50, freq='5T'),
    'CGM': np.random.normal(100, 10, 50),
    'Insulin Activity': np.zeros(50),
    'Food Activity': np.zeros(50),
    'CGM Gradient': np.zeros(50)
})

closed_loop_system = ClosedLoopSystem(predictor)
closed_loop_system.simulate(initial_data)