# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df = pd.read_csv("1553768847-housing.csv")

"""### Some Values Still Empty So Rows Will Be Deleted"""

df.dropna(inplace=True)

"""**DATA IS CLEANED**"""

mean_value = df['median_house_value'].mean()
print("Mean value is: "mean_value)

X = df[['longitude', 'latitude', 'housing_median_age', 'median_income']]
y = df['median_house_value']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

train_predictions = model.predict(X_train)
train_rmse = mean_squared_error(y_train, train_predictions, squared=False)
print(f"Training RMSE: {train_rmse}")

test_predictions = model.predict(X_test)
test_rmse = mean_squared_error(y_test, test_predictions, squared=False)
print(f"Test RMSE: {test_rmse}")
