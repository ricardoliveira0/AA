import NaiveBayesUevora
import numpy as n
import pandas as p

print("\t Teste1")
nba = NaiveBayesUevora.NaiveBayesUevora()
file = p.read_csv("assets/breast-cancer-train.csv")
x = file.drop([file.columns[-1]], axis=1) # drop last column | 1 for column
y = file[file.columns[-1]]
nba.__init__(1)
nba.fit(x, y)

file = p.read_csv("assets/breast-cancer-test.csv")
x1 = file.drop([file.columns[-1]], axis=1) # drop last column | 1 for column
y1 = file[file.columns[-1]]
for q, r in zip(n.array(x1), nba.predict(x1)):
    print(f'{q} --> {r}')
print(f'Precisão: {nba.precision_score(x1, y1)}')
print(f'Exatidão: {nba.accuracy_score(x1, y1)}')
