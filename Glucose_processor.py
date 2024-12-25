import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from scipy.signal import butter, filtfilt

class GlucoseDataProcessor:

    def __init__(self, CGM_file_path: str, bolus_file_path: str,
                 carb_file_path: str, min_range: float,
                 max_range: float, target: float,
                 tp: float, td: float, dp: float, dd: float):
        
        self.CGM_data = pd.read_csv(CGM_file_path)
        self.bolus_data = pd.read_csv(bolus_file_path)
        self.carb_data = pd.read_csv(carb_file_path)
        self.min_range = min_range
        self.max_range = max_range
        self.target = target
        self.tp = tp
        self.td = td
        self.dp = dp
        self.dd = dd


    def _marker_colors(self) -> list:
        """
        Calculate the color for each marker based on its value.
        Green: Within range, closer to target is brighter green.
        Red: Below range, darker red as farther from target.
        Purple: Above range, darker purple as farther from target.
        """
        colors = []
        for value in self.CGM_data['Value']:
            if self.min_range <= value <= self.max_range:
                # In range, interpolate green intensity based on proximity to target
                intensity = 255 - int(abs(value - self.target) / (self.max_range - self.min_range) * 255)
                colors.append(f'rgba(0, {intensity}, 0, 0.5)')  # Shades of green
            elif value < self.min_range:
                # Below range, interpolate red intensity based on distance from min_range
                intensity = 255 - int(abs(value - self.min_range) / abs(0 - self.min_range) * 255)
                colors.append(f'rgba({intensity}, 0, 0, 0.5)')  # Shades of red
            else:
                # Above range, interpolate purple intensity based on distance from max_range
                intensity = 255- int(abs(value - self.max_range) / abs(500 - self.max_range) * 255)
                colors.append(f'rgba({intensity}, 0, {intensity}, 0.5)')  # Shades of purple
        

        return colors
    

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
        t = np.flip(t)

        # All bolus times
        t_0 = np.array(self.bolus_data["Device Time"])
        t_0 = self._calculate_time_differences(start_time=start_time, time_array=t_0)
        t_0 = np.flip(t_0)
    
        # Calculate the time constant and the activation function (NOTE: starts at 0 at t=0)
        tau = self.tp*(1-self.tp/self.td)/(1-2*self.tp/self.td)
        ia = np.zeros(len(t))

        # Create linear combination of the insulin activity curve using the bolus times:
        for t0 in range(len(t_0)):
            for i in range(len(t)):
                if t[i] < t_0[t0]:
                    ia[i] += 0
                else:
                    dia = (self.bolus_data["Normal"].iloc[-(t0+1)]/tau**2)*(t[i]-t_0[t0])*(1-((t[i]-t_0[t0])/self.td))*np.exp(-(t[i]-t_0[t0])/tau)
                    if dia >= 0:
                        ia[i] += dia
                    else:
                        ia[i] += 0

        return ia
    

        # Create the food activity function
    def food_activity_curve(self):

        # Create time domain for the function (same as the CGM data but in mins since start for easier integration)
        start_time = self.CGM_data["Device Time"].iloc[-1]
        t_datetime = np.array(self.CGM_data["Device Time"])
        
        t = self._calculate_time_differences(start_time=start_time, time_array=t_datetime)
        t = np.flip(t)

        # All bolus times
        t_0 = np.array(self.carb_data["Device Time"])
        t_0 = self._calculate_time_differences(start_time=start_time, time_array=t_0)
        t_0 = np.flip(t_0)
    
        # Calculate the time constant and the activation function (NOTE: starts at 0 at t=0)
        tau = self.tp*(1-self.dp/self.dd)/(1-2*self.dp/self.dd)
        fa = np.zeros(len(t))

        # Create linear combination of the insulin activity curve using the bolus times:
        for t0 in range(len(t_0)):
            for i in range(len(t)):
                if t[i] < t_0[t0]:
                    fa[i] += 0
                else:
                    dfa = (self.carb_data["Carb Input"].iloc[-(t0+1)]/tau**2)*(t[i]-t_0[t0])*(1-((t[i]-t_0[t0])/self.dd))*np.exp(-(t[i]-t_0[t0])/tau)
                    if dfa >= 0:
                        fa[i] += dfa
                    else:
                        fa[i] += 0

        return fa



    def plot_data(self):
        """
        Plots the CSV data from a csv file into 3 long plotly scatter plot (CGM data first, then insulin activity)
        Contains scatter and line plots for visibility, and makes use to _marker_colors
        to change the color of the markers
        """

        # Calculate the insulin and food activity
        insulin_activity_data = self.insulin_activity_curve()
        food_activity_data = self.food_activity_curve()

        fig = make_subplots(rows=3, cols=1, vertical_spacing=0.05, shared_xaxes=True)

        # Calculate the color of the markers
        marker_colors = self._marker_colors()

        # CGM line
        fig.add_trace(go.Scatter(
            x=self.CGM_data["Device Time"],
            y=self.CGM_data["Value"],
            mode='lines',
            line=dict(width=3, color='white', shape='spline'),
            opacity=1
        ), row=1, col=1)

        # CGM datapoints of the different colors
        fig.add_trace(go.Scatter(
            x=self.CGM_data["Device Time"],
            y=self.CGM_data["Value"],
            mode='markers',
            marker=dict(size=8, color=marker_colors)
        ), row=1, col=1)

        # Add the lower limit
        fig.add_trace(go.Scatter(
            x=self.CGM_data["Device Time"], 
            y=[self.min_range]*len(self.CGM_data["Device Time"]),
            mode='lines',
            name="Lower Range",
            line=dict(color='green', dash='dash')
        ), row=1, col=1)

        # Add the upper limit
        fig.add_trace(go.Scatter(
            x=self.CGM_data["Device Time"], 
            y=[self.max_range]*len(self.CGM_data["Device Time"]),
            mode='lines',
            name="Upper Range",
            line=dict(color='green', dash='dash')
        ), row=1, col=1)

        # Fill between the two lines
        fig.add_trace(go.Scatter(
            x=self.CGM_data["Device Time"],
            y=[self.min_range]*len(self.CGM_data["Device Time"]),
            mode='none',  # Don't plot the points
            fill='tonexty',  # Fill the area between the two lines
            fillcolor='rgba(0, 255, 0, 0.1)'  # Set the fill color with transparency
        ), row=1, col=1)

        # Insulin and food line
        fig.add_trace(go.Scatter(
            x=self.CGM_data["Device Time"].iloc[::-1],
            y=insulin_activity_data,
            mode='lines',
            line=dict(width=3, color='white', shape='spline'),
            opacity=1
        ), row=2, col=1)

        fig.add_trace(go.Scatter(
            x=self.CGM_data["Device Time"].iloc[::-1],
            y=food_activity_data,
            mode='lines',
            line=dict(width=3, color='blue', shape='spline'),
            opacity=1
        ), row=2, col=1)

        # Gradient
        fig.add_trace(go.Scatter(
            x=self.CGM_data["Device Time"],
            y=self._filter_data(data_name="Value", window=0.06),
            mode='lines',
            line=dict(width=3, color='red', shape='spline'),
            opacity=1
        ), row=1, col=1)

        # Food activity
        fig.add_trace(go.Scatter(
            x=self.CGM_data["Device Time"],
            y=self._calculate_time_gradient(data_name="Value", window=0.06),
            mode='lines',
            line=dict(width=3, color='red', shape='spline'),
            opacity=1
        ), row=3, col=1)


        fig.update_yaxes(fixedrange=True, range=[0,400], row=1, col=1)
        fig.update_yaxes(fixedrange=True, row=2, col=1)

        fig.update_xaxes(fixedrange=False, rangeslider=dict(visible=True, thickness=0.04), row=2, col=1)

        fig.update_layout(dragmode='pan',
                          yaxis=dict(fixedrange=True),
                          height=1300,
                          margin=dict(l=0, r=0, t=0, b=0),
                          autosize=False, showlegend=False)

        return fig
    
    # Find the gradient of the curves
    def _filter_data(self, data_name: str, window: float):

        data=self.CGM_data[f"{data_name}"]

        # Design a Butterworth filter
        cutoff = window  # Normalized cutoff frequency (0 < cutoff < 0.5)
        b, a = butter(N=2, Wn=cutoff, btype='low', analog=False)

        # Apply the filter
        filtered_data = filtfilt(b, a, data)

        return filtered_data



    # Find the gradient of the curves
    def _calculate_time_gradient(self, data_name: str, window: float):

        filtered_data = self._filter_data(data_name=data_name, window=window)

        # Time difference from scratch
        time_difference = self._calculate_time_differences(start_time=self.CGM_data["Device Time"].iloc[-1], time_array=np.array(self.CGM_data["Device Time"]))

        derivative = np.gradient(filtered_data, time_difference)

        return derivative
    
    
    # Find the food activity. this follows the equation d/dt(BG) = k(ia-fa)
    def food_activity(self, k, window: float):

        dBG = np.array(self._calculate_time_gradient(data_name="Value", window=window))
        ia = np.array(self.insulin_activity_curve())

        return dBG/k - ia
    
    def create_DataFrame(self) -> pd.DataFrame:
        CGM_data = self.CGM_data["Value"].iloc[::-1]
        time_data = self.CGM_data["Device Time"].iloc[::-1]
        insulin_activity_data = self.insulin_activity_curve()
        food_activity_data = self.food_activity_curve()

        df = pd.DataFrame({"Time": time_data, "CGM": CGM_data, "Insulin Activity": insulin_activity_data, "Food Activity": food_activity_data})

        return df

    def create_filtered_DataFrame(self) -> pd.DataFrame:
        CGM_data = self.CGM_data["Value"].iloc[::-1]
        filtered_CGM_data = self._filter_data(data_name="Value", window=0.06)
        time_data = self.CGM_data["Device Time"].iloc[::-1]
        insulin_activity_data = self.insulin_activity_curve()
        food_activity_data = self.food_activity_curve()

        df = pd.DataFrame({"Time": time_data, "CGM": filtered_CGM_data, "Insulin Activity": insulin_activity_data, "Food Activity": food_activity_data})

        return df

