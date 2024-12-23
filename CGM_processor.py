import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class CGMDataProcessor:

    def __init__(self, csv_file_path: str, min_range: float, max_range: float, target: float):
        self.data = pd.read_csv(csv_file_path)
        self.min_range = min_range
        self.max_range = max_range
        self.target = target


    def _marker_colors(self) -> list:
        """
        Calculate the color for each marker based on its value.
        Green: Within range, closer to target is brighter green.
        Red: Below range, darker red as farther from target.
        Purple: Above range, darker purple as farther from target.
        """
        colors = []
        for value in self.data['Value']:
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


    def plot_data(self):
        """
        Plots the CSV data from a csv file into a long plotly scatter plot.
        Contains scatter and line plots for visibility, and makes use to _marker_colors
        to change the color of the markers
        """

        fig = make_subplots(rows=3, cols=1, vertical_spacing=0.1, shared_xaxes=True)

        # Calculate the color of the markers
        marker_colors = self._marker_colors()

        # CGM line
        fig.add_trace(go.Scatter(
            x=self.data["Device Time"],
            y=self.data["Value"],
            mode='lines',
            line=dict(width=3, color='white', shape='spline'),
            opacity=1
        ), row=1, col=1)

        # CGM datapoints of the different colors
        fig.add_trace(go.Scatter(
            x=self.data["Device Time"],
            y=self.data["Value"],
            mode='markers',
            marker=dict(size=8, color=marker_colors)
        ), row=1, col=1)

        # Add the lower limit
        fig.add_trace(go.Scatter(
            x=self.data["Device Time"], 
            y=[self.min_range]*len(self.data["Device Time"]),
            mode='lines',
            name="Lower Range",
            line=dict(color='green', dash='dash')
        ), row=1, col=1)

        # Add the upper limit
        fig.add_trace(go.Scatter(
            x=self.data["Device Time"], 
            y=[self.max_range]*len(self.data["Device Time"]),
            mode='lines',
            name="Upper Range",
            line=dict(color='green', dash='dash')
        ), row=1, col=1)

        # Fill between the two lines
        fig.add_trace(go.Scatter(
            x=self.data["Device Time"],
            y=[self.min_range]*len(self.data["Device Time"]),
            mode='none',  # Don't plot the points
            fill='tonexty',  # Fill the area between the two lines
            fillcolor='rgba(0, 255, 0, 0.1)'  # Set the fill color with transparency
        ), row=1, col=1)

        # CGM line
        fig.add_trace(go.Scatter(
            x=self.data["Device Time"],
            y=self.data["Value"],
            mode='lines',
            line=dict(width=3, color='white', shape='spline'),
            opacity=1
        ), row=2, col=1)

        fig.update_yaxes(range=[0,400], row=1, col=1)
        fig.update_yaxes(range=[0,400], row=2, col=1)

        fig.update_xaxes(fixedrange=False, rangeslider=dict(visible=True), row=2, col=1)

        fig.update_layout(dragmode="pan", height=1300,
                          margin=dict(l=0, r=0, t=0, b=0),
                          autosize=False, showlegend=False)

        return fig