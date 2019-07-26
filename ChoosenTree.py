from sklearn import tree

myTree = tree.DecisionTreeClassifier()

dacTrung = [[1, 3, 3, 7],
            [5, 2, 4, 6],
            [1, 2, 4, 6],
            [5, 4, 4, 3],
            [1, 4, 4, 7],
            [3, 2, 3, 7],
            [3, 3, 3, 6],
            [5, 2, 2, 7]
            ]

nhan = [0, 1, 1, 0, 0, 0, 0, 1]

myTree.fit(dacTrung, nhan)

result = myTree.predict([[1, 3, 3, 7],
                          [5, 2, 4, 6],
                          [1, 2, 4, 6],
                          [5, 4, 4, 3],
                          [1, 4, 4, 7],
                          [3, 2, 3, 7],
                          [3, 3, 3, 6],
                          [5, 2, 2, 7]
                          ])
print(result)