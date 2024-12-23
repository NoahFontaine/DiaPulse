import pandas as pd
import numpy as np
from datetime import datetime

class IADataProcessor:

    def __init__(self, CGM_file_path: str, bolus_file_path: str, tp: float, td: float):
        self.CGM_data = pd.read_csv(CGM_file_path)
        self.bolus_data = pd.read_csv(bolus_file_path)
        self.tp = tp
        self.td = td

    def _calculate_time_differences(self, start_time, time_array):
        """
        Calculate time differences in minutes between a start time and an array of times.

        Parameters:
        - start_time (str): The start time in 'yyyy-mm-dd hh:mm:ss' format.
        - time_array (np.ndarray): Array of times in 'yyyy-mm-dd hh:mm:ss' format.

        Returns:
        - np.ndarray: Array of time differences in minutes.
        """
        # Convert start_time to a datetime object
        start_time_dt = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
        
        # Convert each time in the array to a datetime object and calculate the difference
        time_differences = [
            (datetime.strptime(t, '%Y-%m-%d %H:%M:%S') - start_time_dt).total_seconds() / 60
            for t in time_array
        ]
        
        return np.array(time_differences)


    # Create the insulin activity function
    def insulin_activity_curve(self):

        # Create time domain for the function (same as the CGM data but in mins since start for easier integration)
        start_time = self.CGM_data["Device Time"].iloc[-1]
        t_datetime = np.array(self.CGM_data["Device Time"])
        
        t = self._calculate_time_differences(start_time=start_time, time_array=t_datetime)

        print(t)

        # All bolus times
        t_0 = np.array(self.bolus_data["Device Time"])
        t_0 = self._calculate_time_differences(start_time=start_time, time_array=t_0)

        print(t_0)
    
        # Calculate the time constant and the activation function (NOTE: starts at 0)
        tau = self.tp*(1-self.tp/self.td)/(1-2*self.tp/self.td)
        ia = 0

        # Create linear combination of the insulin activity curve using the bolus times:
        for t0 in range(len(t_0)):
            ia += (self.bolus_data["Normal"][t0]/tau**2)*(t-t_0[t0])*(1-(t-t_0[t0])/self.td)*np.exp(-(t-t_0[t0])/tau)

        print(len(ia))
        print(len(self.CGM_data["Device Time"]))

        return ia