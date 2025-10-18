# Ex.No: 07                                       AUTO REGRESSIVE MODEL
### Date: 4-10-2025



### AIM:
To Implementat an Auto Regressive Model using Python
### ALGORITHM:
1. Import necessary libraries
2. Read the CSV file into a DataFrame
3. Perform Augmented Dickey-Fuller test
4. Split the data into training and testing sets.Fit an AutoRegressive (AR) model with 13 lags
5. Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF)
6. Make predictions using the AR model.Compare the predictions with the test data
7. Calculate Mean Squared Error (MSE).Plot the test data and predictions.
### PROGRAM

```py
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error

# -----------------------------
# Load the dataset
# -----------------------------
data = pd.read_csv('/content/avocado.csv', parse_dates=['Date'], index_col='Date')
data = data.sort_index()
# Use Total Volume
print("Columns:", data.columns)
print("\nUsing column: 'Total Volume'\n")

# -----------------------------
# Make data stationary (differencing)
# -----------------------------
data['Total_Volume_Diff'] = data['Total Volume'].diff()
data = data.dropna()  # remove NaN from differencing

# ADF Test
result = adfuller(data['Total_Volume_Diff'])
print('ADF Statistic:', result[0])
print('p-value:', result[1])
if result[1] > 0.05:
    print("Series is likely NON-stationary\n")
else:
    print("Series is likely STATIONARY\n")

# -----------------------------
# Train-test split
# -----------------------------
x = int(0.8 * len(data))
train_data = data.iloc[:x]
test_data = data.iloc[x:]

# -----------------------------
# Fit AR Model on differenced data
# -----------------------------
lag_order = 13
model = AutoReg(train_data['Total_Volume_Diff'], lags=lag_order)
model_fit = model.fit()

# -----------------------------
# ACF and PACF plots
# -----------------------------
plt.figure(figsize=(10, 6))
plot_acf(data['Total_Volume_Diff'], lags=40, alpha=0.05)
plt.title('Autocorrelation Function (ACF) - Differenced Total Volume')
plt.show()

plt.figure(figsize=(10, 6))
plot_pacf(data['Total_Volume_Diff'], lags=40, alpha=0.05)
plt.title('Partial Autocorrelation Function (PACF) - Differenced Total Volume')
plt.show()

# -----------------------------
# Predictions
# -----------------------------
predictions = model_fit.predict(
    start=len(train_data),
    end=len(train_data) + len(test_data) - 1
)

# Accuracy
mse = mean_squared_error(test_data['Total_Volume_Diff'], predictions)
print('Mean Squared Error (MSE):', mse)

# -----------------------------
# Plot Predictions vs Test Data
# -----------------------------
plt.figure(figsize=(12, 6))
plt.plot(test_data['Total_Volume_Diff'], label='Test Data (Differenced)')
plt.plot(predictions, label='Predictions - AR Model', linestyle='--')
plt

```
### OUTPUT:

GIVEN DATA
<img width="873" height="144" alt="image" src="https://github.com/user-attachments/assets/d6764eb3-e5aa-4136-bbac-39b8a5a64a03" />

ADF Statistic
<img width="355" height="92" alt="image" src="https://github.com/user-attachments/assets/402b7bf9-522c-4d25-90d3-32d260de93ba" />

ACF :
<img width="651" height="493" alt="image" src="https://github.com/user-attachments/assets/7ae38c99-795e-497a-b8ae-c6b885fea8f3" />

PACF:
<img width="666" height="486" alt="image" src="https://github.com/user-attachments/assets/f470eefd-846a-43dd-bb03-10dcab730f31" />

MSE:
<img width="393" height="17" alt="image" src="https://github.com/user-attachments/assets/8e1701e3-9d63-4679-ab46-55356b98a3cd" />


FINIAL PREDICTION
<img width="902" height="462" alt="image" src="https://github.com/user-attachments/assets/0ca5c5ba-46ba-43c4-9994-e0d38f771987" />

### RESULT:
Thus we have successfully implemented the auto regression function using python.
