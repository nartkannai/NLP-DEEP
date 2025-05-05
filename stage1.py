import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

# Load data
df = pd.read_excel("pp.xlsx", sheet_name="Sheet1")
df.dropna(axis=0, inplace=True)

# Split data into features and target
X = df.iloc[:, :4].values
y = df.iloc[:, 4].values

# Normalize data
scaler_X = MinMaxScaler(feature_range=(0, 1))
scaler_y = MinMaxScaler(feature_range=(0, 0.9))

X = scaler_X.fit_transform(X)
y = np.reshape(y, (-1, 1))
y = scaler_y.fit_transform(y)

# Split data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.33, random_state=42)

# Build the model
model = Sequential()
model.add(Dense(units=4, activation='tanh', input_dim=4))
model.add(Dense(units=1, activation='linear'))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, epochs=1000, batch_size=64, validation_data=(X_valid, y_valid), verbose=0)

# Evaluate the model on the test set
y_test_pred = model.predict(X_test)
y_test_pred = scaler_y.inverse_transform(y_test_pred)
y_test = scaler_y.inverse_transform(y_test)

# Calculate MAPE for the test set
mape_test = np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100
print(f"MAPEtest = {mape_test}")

# Plot training history
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Convergence History')
plt.legend()
plt.show()
