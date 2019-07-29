from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris["data"], iris["target"], random_state=0)
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
print("result\n")
print(model.predict(X_test))
print("y_test\n")
print(y_test)
print(model.score(X_test,y_test))    