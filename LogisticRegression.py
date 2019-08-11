import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("dataclass.csv", header=None)
X = df.values[:, [0,1]]
y = df.values[:, 2]
X_train, X_test, y_train, y_test = train_test_split(X, y)

for i in range(len(X)):
    plt.scatter(X[i:, 0], X[i:, 1], c=("b" if y[i] == 1 else "r"))
    
plt.axis([-2, 12, -2, 12])

plt.show()

model = LogisticRegression()
model.fit(X_train, y_train)
result = model.predict(X_test)

print(result)
print(y_test)
print(model.score(X_test,y_test))