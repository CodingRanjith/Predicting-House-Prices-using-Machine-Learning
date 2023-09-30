# Required Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import numpy as np

# 1. Data Loading
# Assuming you have a dataset named 'house_prices.csv'
data = pd.read_csv('house_prices.csv')

# 2. Data Preprocessing
# Fill missing values for simplicity (better methods might be needed based on the dataset)
data.fillna(method='ffill', inplace=True)

# Convert categorical features using get_dummies (one-hot encoding)
data = pd.get_dummies(data, drop_first=True)

# 3. Feature Selection (For simplicity, we'll use all features to predict the price)
X = data.drop('price', axis=1)
y = data['price']

# 4. Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 5. Model Selection & Training
model = LinearRegression()
model.fit(X_train, y_train)

# 6. Evaluation
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print(f"MAE: {mae}")
print(f"RMSE: {rmse}")
print(f"R2 Score: {r2}")
