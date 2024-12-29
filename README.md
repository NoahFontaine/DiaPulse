# DiaPulse

Trying to predict what my Blood Glucose level (BG) will be in 5-120mins - in 5mins intervals - based of CGM and bolus data from my pump.

View training and results of the models using the training_{#}.ipynb notebooks.

## Models

### <span style="color:blue">Model A:</span>

Simple Sequential model with 1 LSTM hidden layer.

Features are:
 - Time of day (both sine and cosine components).
 - Current CGM BG data.
 - Current "Insulin Activity" - pseudo $f(active insulin, insulin characteristics, t)$ to try and model how insulin gets absorbed into the body.
 - Current "Food Activity" - effectively the same type of function as Insulin Activity, but with different parameters to try and model how food gets digested.

#### <span style="color:blue">Architecture:</span>

Input (5) -> LSTM (64) -> Output (24)


### <span style="color:blue">Model B:</span>

Added new lagging feature and a second LSTM hidden layer, with 2 dropout layers too.

Extra features are:
 - Time of day 5, 10, 15, 20 mins before (both sine and cosine components).

#### <span style="color:blue">Architecture:</span>

Input (9) -> Dropout -> LSTM (64) -> Dropout -> LSTM (32) -> Batch Normalization -> Dropout -> Output (24)

### <span style="color:blue">Loss and Residuals:</span>

Test Loss: 0.0037, Test MAE: 0.0420

| Horizon (minutes) | Mean | Standard Deviation |
|--------------------|---------------------------|--------------------------------|
| Horizon 5 mins: | Mean=2.69, | Std=6.69
| Horizon 10 mins: | Mean=2.23, | Std=6.14
| Horizon 15 mins: | Mean=1.94, | Std=5.93
| Horizon 20 mins: | Mean=1.35, | Std=6.11
| Horizon 25 mins: | Mean=1.07, | Std=6.77
| Horizon 30 mins: | Mean=0.79, | Std=7.78
| Horizon 35 mins: | Mean=0.49, | Std=9.10
| Horizon 40 mins: | Mean=0.26, | Std=10.60
| Horizon 45 mins: | Mean=0.21, | Std=12.23
| Horizon 50 mins: | Mean=0.27, | Std=13.92
| Horizon 55 mins: | Mean=0.35, | Std=15.64
| Horizon 60 mins: | Mean=0.39, | Std=17.33
| Horizon 65 mins: | Mean=0.58, | Std=18.98
| Horizon 70 mins: | Mean=0.78, | Std=20.57
| Horizon 75 mins: | Mean=0.94, | Std=22.08
| Horizon 80 mins: | Mean=1.05, | Std=23.50
| Horizon 85 mins: | Mean=1.27, | Std=24.82
| Horizon 90 mins: | Mean=1.69, | Std=26.06
| Horizon 95 mins: | Mean=1.95, | Std=27.20
| Horizon 100 mins: | Mean=2.13, | Std=28.28
| Horizon 105 mins: | Mean=2.63, | Std=29.30
| Horizon 110 mins: | Mean=2.89, | Std=30.27
| Horizon 115 mins: | Mean=3.13, | Std=31.22
| Horizon 120 mins: | Mean=3.54, | Std=32.16


### <span style="color:blue">Model C:</span>

Same as Model B but with an attention layer after the first LSTM and Layer instead of Batch Normalization. Have tried adding FFT features without success. Will try to add day of week/weekend features and increase the time lag features to 30mins before.

#### <span style="color:blue">Architecture:</span>

Input (9) -> LSTM (64) -> Attention -> LSTM (32) -> Layer Normalization -> Dropout -> Output (24)

#### <span style="color:blue">Loss and Residuals:</span>

Test Loss: 0.0025, Test MAE: 0.0335

| Horizon (minutes) | Mean | Standard Deviation |
|--------------------|---------------------------|--------------------------------|
| Horizon 5 mins: | Mean=0.97, | Std=4.13
| Horizon 10 mins: | Mean=1.01, | Std=3.79
| Horizon 15 mins: | Mean=1.04, | Std=3.99
| Horizon 20 mins: | Mean=1.04, | Std=4.71
| Horizon 25 mins: | Mean=1.01, | Std=5.79
| Horizon 30 mins: | Mean=0.94, | Std=7.09
| Horizon 35 mins: | Mean=0.85, | Std=8.51
| Horizon 40 mins: | Mean=0.72, | Std=9.98
| Horizon 45 mins: | Mean=0.57, | Std=11.45
| Horizon 50 mins: | Mean=0.40, | Std=12.88
| Horizon 55 mins: | Mean=0.22, | Std=14.25
| Horizon 60 mins: | Mean=0.02, | Std=15.52
| Horizon 65 mins: | Mean=-0.18, | Std=16.69
| Horizon 70 mins: | Mean=-0.38, | Std=17.75
| Horizon 75 mins: | Mean=-0.57, | Std=18.71
| Horizon 80 mins: | Mean=-0.76, | Std=19.59
| Horizon 85 mins: | Mean=-0.93, | Std=20.39
| Horizon 90 mins: | Mean=-1.09, | Std=21.15
| Horizon 95 mins: | Mean=-1.23, | Std=21.90
| Horizon 100 mins: | Mean=-1.35, | Std=22.65
| Horizon 105 mins: | Mean=-1.44, | Std=23.43
| Horizon 110 mins: | Mean=-1.51, | Std=24.26
| Horizon 115 mins: | Mean=-1.55, | Std=25.15
| Horizon 120 mins: | Mean=-1.56, | Std=26.11

### <span style="color:blue">Model D:</span>

Same as Model C but with more lag (6 steps ahead) on CGM data and added lag features for insulin and food activity.

#### <span style="color:blue">Architecture:</span>

Input (9) -> LSTM (64) -> Attention -> LSTM (32) -> Layer Normalization -> Dropout -> Output (24)

#### <span style="color:blue">Loss and Residuals:</span>

Test Loss: 0.0024, Test MAE: 0.0339

| Horizon (minutes) | Mean | Standard Deviation |
|--------------------|---------------------------|--------------------------------|
|Horizon 5 mins: | Mean=0.62, | Std=4.61
|Horizon 10 mins: | Mean=0.54, | Std=4.47
|Horizon 15 mins: | Mean=0.44, | Std=4.81
|Horizon 20 mins: | Mean=0.33, | Std=5.56
|Horizon 25 mins: | Mean=0.21, | Std=6.61
|Horizon 30 mins: | Mean=0.09, | Std=7.85
|Horizon 35 mins: | Mean=-0.04, | Std=9.19
|Horizon 40 mins: | Mean=-0.16, | Std=10.56
|Horizon 45 mins: | Mean=-0.28, | Std=11.93
|Horizon 50 mins: | Mean=-0.39, | Std=13.26
|Horizon 55 mins: | Mean=-0.49, | Std=14.50
|Horizon 60 mins: | Mean=-0.57, | Std=15.65
|Horizon 65 mins: | Mean=-0.64, | Std=16.69
|Horizon 70 mins: | Mean=-0.69, | Std=17.62
|Horizon 75 mins: | Mean=-0.70, | Std=18.43
|Horizon 80 mins: | Mean=-0.70, | Std=19.15
|Horizon 85 mins: | Mean=-0.67, | Std=19.80
|Horizon 90 mins: | Mean=-0.60, | Std=20.41
|Horizon 95 mins: | Mean=-0.52, | Std=21.00
|Horizon 100 mins: | Mean=-0.40, | Std=21.61
|Horizon 105 mins: | Mean=-0.26, | Std=22.27
|Horizon 110 mins: | Mean=-0.07, | Std=22.99
|Horizon 115 mins: | Mean=0.13, | Std=23.82
|Horizon 120 mins: | Mean=0.39, | Std=24.74

### <span style="color:blue">Model E:</span>

Same as Model D but with added CGM gradient and gradient lag features.

#### <span style="color:blue">Architecture:</span>

Input (9) -> LSTM (64) -> Attention -> LSTM (32) -> Layer Normalization -> Dropout -> Output (24)

#### <span style="color:blue">Loss and Residuals:</span>

Test Loss: 0.0012, Test MAE: 0.0226

| Horizon (minutes) | Mean | Standard Deviation |
|--------------------|---------------------------|--------------------------------|
|Horizon 5 mins: | Mean=1.72, | Std=3.62
|Horizon 10 mins: | Mean=1.61, | Std=2.96
|Horizon 15 mins: | Mean=1.49, | Std=2.52
|Horizon 20 mins: | Mean=1.37, | Std=2.35
|Horizon 25 mins: | Mean=1.24, | Std=2.50
|Horizon 30 mins: | Mean=1.10, | Std=2.97
|Horizon 35 mins: | Mean=0.95, | Std=3.69
|Horizon 40 mins: | Mean=0.79, | Std=4.58
|Horizon 45 mins: | Mean=0.62, | Std=5.60
|Horizon 50 mins: | Mean=0.44, | Std=6.69
|Horizon 55 mins: | Mean=0.25, | Std=7.82
|Horizon 60 mins: | Mean=0.07, | Std=8.93
|Horizon 65 mins: | Mean=-0.11, | Std=10.01
|Horizon 70 mins: | Mean=-0.28, | Std=11.04
|Horizon 75 mins: | Mean=-0.42, | Std=12.00
|Horizon 80 mins: | Mean=-0.55, | Std=12.90
|Horizon 85 mins: | Mean=-0.64, | Std=13.74
|Horizon 90 mins: | Mean=-0.71, | Std=14.55
|Horizon 95 mins: | Mean=-0.75, | Std=15.34
|Horizon 100 mins: | Mean=-0.76, | Std=16.15
|Horizon 105 mins: | Mean=-0.73, | Std=17.00
|Horizon 110 mins: | Mean=-0.69, | Std=17.92
|Horizon 115 mins: | Mean=-0.61, | Std=18.94
|Horizon 120 mins: | Mean=-0.50, | Std=20.07