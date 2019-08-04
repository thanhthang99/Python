from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston

X = [[1.1], [2.0], [3.0], [4.0], [5.0]]
y = [3.0, 5.1, 6.9, 9.1, 11.0]


model = LinearRegression()
model.fit(X, y)

test=[[2.5]]
print(model.predict(test))
print(model.coef_)
print(model.intercept_)