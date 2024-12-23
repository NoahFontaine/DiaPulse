import pandas as pd

data = pd.read_csv('Bolus_data.csv')

# Keep only "CGM" and "timezone" columns
df = data.loc[:, ["Device Time",
                  "Sub Type",
                  "Normal", "Expected Normal",
                  "Duration (mins)", "Expected Duration (mins)",
                  "Extended", "Expected Extended"]]

df.to_csv('Bolus_data.csv')