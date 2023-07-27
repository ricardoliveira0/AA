import NaiveBayesUevora
import numpy as n
import pandas as p
alpha = 1

print("\t Teste1")
nba = NaiveBayesUevora.NaiveBayesUevora()
file = p.read_csv("assets/breast-cancer-train.csv")
x = file.drop([file.columns[-1]], axis=1) # drop last column | 1 for column
y = file[file.columns[-1]]
nba.__init__(alpha)
nba.fit(x, y) # gerar o classificador

file = p.read_csv("assets/breast-cancer-test.csv")
x1 = file.drop([file.columns[-1]], axis=1) # drop last column | 1 for column
y1 = file[file.columns[-1]]
for q, r in zip(n.array(x1), nba.predict(x1)):
    print(f'{q} --> {r}')
print(f'Precisão: {nba.precision_score(x1, y1)}')
print(f'Exatidão: {nba.accuracy_score(x1, y1)}')


print("\t Teste2")
nba = NaiveBayesUevora.NaiveBayesUevora()
file = p.read_csv("assets/breast-cancer-train2.csv")
x = file.drop([file.columns[-1]], axis=1) # drop last column | 1 for column
y = file[file.columns[-1]]
nba.__init__(alpha)
nba.fit(x, y)

file = p.read_csv("assets/breast-cancer-test2.csv")
x1 = file.drop([file.columns[-1]], axis=1) # drop last column | 1 for column
y1 = file[file.columns[-1]]
for q, r in zip(n.array(x1), nba.predict(x1)):
    print(f'{q} --> {r}')
print(f'Precisão: {nba.precision_score(x1, y1)}')
print(f'Exatidão: {nba.accuracy_score(x1, y1)}')

print("\t Teste3")
nba = NaiveBayesUevora.NaiveBayesUevora()
file = p.read_csv("assets/weather-nominal.csv")
x = file.drop([file.columns[-1]], axis=1) # drop last column | 1 for column
y = file[file.columns[-1]]
nba.__init__(alpha)
nba.fit(x, y)

x1 = [["overcast", "mild", "high", "TRUE"]]
print("Previsão:")
for q, r in zip(n.array(x1), nba.predict(x1)):
    print(f'{q} --> {r}')
