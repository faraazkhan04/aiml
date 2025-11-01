import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Step 1: Load the data
df = pd.read_excel("housing_prices_SLR.xlsx")
df.head()

# Step 2: Define features (X) and target (Y)
x = df[['AREA']].values   # Feature matrix
y = df.PRICE.values       # Target variable
x[:5]
y[:5]

# Step 3: Split data into training and testing sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

# Step 4: Train the Linear Regression model
from sklearn.linear_model import LinearRegression
lr_model = LinearRegression()
lr_model.fit(x_train, y_train)

print("Intercept (b0):", lr_model.intercept_)
print("Coefficient (b1):", lr_model.coef_)

# Step 5: Train another model without intercept
lr_model_no_intercept = LinearRegression(fit_intercept=False)
lr_model_no_intercept.fit(x_train, y_train)

print("Intercept (b0):", lr_model_no_intercept.intercept_)
print("Coefficient (b1):", lr_model_no_intercept.coef_)

# Step 6: Calculate R² score
from sklearn.metrics import r2_score
y_train
lr_model.predict(x_train)
r2_score(y_train, lr_model.predict(x_train))
r2_score(y_test, lr_model.predict(x_test))

# Step 7: Visualization
plt.scatter(x_train[:,0], y_train, color='red', label='Training Data')
plt.scatter(x_test[:,0], y_test, color='blue', label='Testing Data')
plt.plot(x_train[:,0], lr_model.predict(x_train), color='yellow', label='Regression Line')
plt.legend()
plt.show() 
