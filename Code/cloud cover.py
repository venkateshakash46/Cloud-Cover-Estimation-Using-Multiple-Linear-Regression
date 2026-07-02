import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import seaborn as sns


np.random.seed(42)


try:
    data = pd.read_excel("D:/Arise/Chennai dataset.xlsx")
except FileNotFoundError:
    print("Error: File not found at the specified path. Please check the file location.")
    exit()

# Feature engineering
data['t'] = data['temp']
data['h'] = data['humidity']
data['vis'] = data['visibility']
data['slp'] = data['sealevelpressure']
data['uv'] = data['uvindex']
data['ws'] = data['windspeed']
data['dir'] = data['winddir']
data['solar'] = data['solarradiation']
data['solar_energy'] = data['solarenergy']
data['solar_ratio'] = data['solar'] / (data['solar'].max() + 1e-6)
data['dew_diff'] = data['temp'] - data['dew']
data['temp_range'] = data['tempmax'] - data['tempmin']

# Interaction terms
data['i1'] = data['t'] * data['h']
data['i2'] = data['h'] * data['vis']
data['i3'] = data['t'] * data['vis']
data['i4'] = data['t'] * data['h'] * data['vis']

# Features and target
features = ['t', 'h', 'vis', 'slp', 'uv', 'ws', 'i1', 'i2', 'i3', 'i4', 'temp_range', 'solar_ratio', 'dew_diff']
X = data[features].values
y = data['cloudcover'].values

# Intercept and normalization (added epsilon to avoid division by zero for constant features)
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X_norm = (X - X_mean) / (X_std + 1e-6)  # Epsilon prevents division-by-zero
X_final = np.column_stack([np.ones(len(X_norm)), X_norm])

# Train and test split
n = len(X_final)
split = int(0.8 * n)
idx = np.random.permutation(n)
train_idx, test_idx = idx[:split], idx[split:]

X_train, X_test = X_final[train_idx], X_final[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# Least squares regression
coef, residuals, rank, s = np.linalg.lstsq(X_train, y_train, rcond=None)

# Error evaluation (added MAE for robustness)
y_pred = X_test @ coef
y_pred = np.clip(y_pred, 0, 100)  # Clip predictions to 0-100 range for cloud cover
ss_res = np.sum((y_test - y_pred)**2)
ss_tot = np.sum((y_test - np.mean(y_test))**2)
r2 = 1 - ss_res / ss_tot
rmse = np.sqrt(np.mean((y_test - y_pred)**2))
mae = np.mean(np.abs(y_test - y_pred))  # Mean Absolute Error

print("\n===== MODEL PERFORMANCE =====")
print(f"R² Score : {r2:.4f}")
print(f"RMSE     : {rmse:.4f}")
print(f"MAE      : {mae:.4f}")

# Actual vs predicted
compare_df = pd.DataFrame({
    'Actual CloudCover': y_test,
    'Predicted CloudCover': y_pred,
    'Error (Actual - Predicted)': y_test - y_pred
})

print("\n===== SAMPLE COMPARISON =====")
print(compare_df.head(20))

# Error analysis
errors = y_test - y_pred
mean_error = np.mean(errors)
std_error = np.std(errors)

print(f"\n===== ERROR ANALYSIS =====")
print(f"Mean Error (Bias): {mean_error:.3f}")
print(f"Standard Deviation of Error: {std_error:.3f}")
print(f"Min Error: {errors.min():.3f}")
print(f"Max Error: {errors.max():.3f}")
print("Coefficients:", coef)

# Reference dataframe
error_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred,
    'Error': errors
})

# Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(data[features + ['cloudcover']].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Between Features and Cloud Cover")
plt.show()

# Feature importance
coef_series = pd.Series(coef[1:], index=features).sort_values()
plt.figure(figsize=(10, 6))
coef_series.plot(kind='barh', color='teal')
plt.title("Feature Importance (Linear Coefficients)")
plt.xlabel("Coefficient Value")
plt.ylabel("Features")
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# Error histogram
plt.figure(figsize=(8, 5))
plt.hist(errors, bins=25, color='skyblue', edgecolor='black', alpha=0.8)
plt.axvline(0, color='red', linestyle='--', linewidth=2, label='Error = 0')
plt.axvline(mean_error, color='green', linestyle='-', linewidth=2, label=f'Mean Error = {mean_error:.2f}')
plt.title('Distribution of Prediction Errors')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# Scatter plot
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Cloud Cover")
plt.ylabel("Predicted Cloud Cover")
plt.title(f"Actual vs Predicted Cloud Cover (R²={r2:.2f})")
plt.grid(True)
plt.show()

# Symbolic differentiation (fixed coefficient assignment)
t, h, vis, slp, uv, ws, i1, i2, i3, i4, temp_range, solar_ratio, dew_diff = sp.symbols(
    't h vis slp uv ws i1 i2 i3 i4 temp_range solar_ratio dew_diff'
)

a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13 = sp.symbols(
    'a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13'
)

# Fixed assignment: Now correctly assigns a0 to intercept, etc.
a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13 = coef

CC = (
    a1 * t
    + a2 * h
    + a3 * vis
    + a4 * slp
    + a5 * uv
    + a6 * ws
    + a7 * i1
    + a8 * i2
    + a9 * i3
    + a10 * i4
    + a11 * temp_range
    + a12 * solar_ratio
    + a13 * dew_diff
    + a0
)

dcc_dt = sp.diff(CC, t)
dcc_dh = sp.diff(CC, h)

print("dCC/∂t =", dcc_dt)
print("dCC/∂h =", dcc_dh)

# Time series plot (improved with dual y-axes for different scales)
data['datetime'] = pd.to_datetime(data['datetime'])
fig, ax1 = plt.subplots(figsize=(12, 5))
ax1.plot(data['datetime'], data['cloudcover'], label='Cloud Cover', color='gray')
ax1.set_xlabel("Date")
ax1.set_ylabel("Cloud Cover", color='gray')
ax1.tick_params(axis='y', labelcolor='gray')

ax2 = ax1.twinx()
ax2.plot(data['datetime'], data['humidity'], label='Humidity', color='blue', alpha=0.7)
ax2.set_ylabel("Humidity", color='blue')
ax2.tick_params(axis='y', labelcolor='blue')

fig.suptitle("Cloud Cover and Humidity Over Time")
fig.legend(loc="upper left")
plt.show()
