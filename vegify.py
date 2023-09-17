import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv('veggie.csv')
data['Date'] = pd.to_datetime(data['Date'])
data['Days'] = (data['Date'] - data['Date'].min()).dt.days

X = data[['Days']]
y_carrots = data['Carrot']

X_train, X_test, y_train, y_test = train_test_split(X, y_carrots, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

X_test['Days'] = (X_test['Days'] - X_train['Days'].min())
carrots_price_predictions = model.predict(X_test)

plt.figure(figsize=(10, 6))
plt.scatter(X_test['Days'], y_test, color='blue', label='Actual Prices')
plt.plot(X_test['Days'], carrots_price_predictions, color='red', label='Predicted Prices')
plt.title('Carrots Price Prediction')
plt.xlabel('Days')
plt.ylabel('Price (₹)')
plt.legend()
plt.show()