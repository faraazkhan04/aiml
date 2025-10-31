import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_excel("housing_prices.xlsx")
print(df.head())

# Encode string data (if any)
le = LabelEncoder()
df.iloc[:, 3] = le.fit_transform(df.iloc[:, 3])  # target column

# Features and target
x = df.iloc[:, :3].values
y = df.iloc[:, 3].values

# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)

# Fit model
mlr_model = LinearRegression()
mlr_model.fit(x_train, y_train)

# Output
print("Intercept:", mlr_model.intercept_)
print("Coefficients:", mlr_model.coef_)
print("Training R²:", mlr_model.score(x_train, y_train))
print("Testing R²:", mlr_model.score(x_test, y_test))
