# House Price Prediction using Pandas and Linear Regression

# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Create sample dataset
data = {
    'Area': [1000, 1500, 1800, 2400, 3000],
    'Bedrooms': [2, 3, 3, 4, 5],
    'Age': [10, 5, 8, 2, 1],
    'Price': [200000, 300000, 350000, 500000, 650000]
}

# Convert data into pandas DataFrame
df = pd.DataFrame(data)

# Display dataset
print("Dataset:")
print(df)

# Features (X) and Target (y)
X = df[['Area', 'Bedrooms', 'Age']]
y = df['Price']

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Linear Regression model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Predict house prices
predictions = model.predict(X_test)

# Compare actual vs predicted
result = pd.DataFrame({
    'Actual Price': y_test,
    'Predicted Price': predictions
})

print("\nPrediction Results:")
print(result)

# Model accuracy using Mean Absolute Error
mae = mean_absolute_error(y_test, predictions)
print("\nMean Absolute Error:", mae)

# Predict price for new house
new_house = pd.DataFrame({
    'Area': [2000],
    'Bedrooms': [3],
    'Age': [4]
})

predicted_price = model.predict(new_house)

print("\nPredicted Price for New House:", predicted_price[0])