from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("ADS.csv", header=0)
X = df.values[:, 2]
y = df.values[:, 4]

X_train, y_train, X_test, y_test = train_test_split(X, y, random_state=0)
#plt.scatter(X,y)
#plt.show()

def predict(radio, weight, bias):
    return (weight * radio + bias)

def cost(X, y, weight, bias):
    N = len(X)
    sum = 0.0
    for i in range(N):
        sum += (y[i] - (weight * X[i] + bias))**2
    return (sum / N)

def update(X, y, weight, bias, learning_rate):
    N = len(X)
    weight_temp = 0.0
    bias_temp = 0.0
    for i in range(N):
        weight_temp += - 2 * X[i] * (y[i] - (weight * X[i] + bias))
        bias_temp += - 2 * (y[i] - (weight * X[i] + bias))
    weight -= (weight_temp / N) * learning_rate
    bias -= (bias_temp / N) * learning_rate
    return weight, bias

def train(X, y, weight, bias, learning_rate, iterator):
    costList = []
    for i in range(iterator):
        weight, bias = update(X, y, weight, bias, learning_rate)
        costCurent = cost(X, y, weight, bias)
        costList.append(costCurent)
    return weight, bias, costList

weight, bias, costList = train(X, y, 0.03, 0.0014, 0.001, 60)
print(weight)
print(bias)
print(costList)
print(predict(19, weight, bias))
'''
solanlap = []
for i in range(60):
    solanlap.append(i)
plt.plot(solanlap, costList)
plt.show()
'''
