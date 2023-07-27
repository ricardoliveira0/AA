import pandas as p
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import  GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier


def dataHandling(X):
    y ={}
    y['Id'] = X['Id']
    y['Program'] = X['Program']
    y['media'] = (X['Y1s1_grade']+X['Y1s2_grade']+ \
                  X['Y2s1_grade']+X['Y2s2_grade']+ \
                  X['Y3s1_grade']+X['Y3s2_grade']+ \
                  X['Y4s1_grade']+X['Y4s2_grade'])/8
    return p.DataFrame.from_dict(y,orient='columns')
fileName = "assets/dropout-trabalho2.csv"

# KNN
class modelo:
    @classmethod
    def predict(cls, xTest):
       fileContent = p.read_csv(fileName)
       print(f'{fileContent}')
       X = fileContent.drop('Failure',axis='columns')
       y = fileContent['Failure']
       clf = KNeighborsClassifier(n_neighbors=10)
       clf.fit(X,y)
       return clf.predict(xTest)

# Naïve Bayes
class modelo2:
    @classmethod
    def predict(cls, xTest):
        fileContent = p.read_csv(fileName)
        X = fileContent.drop('Failure', axis='columns')
        y = fileContent['Failure']
        clf = GaussianNB()
        clf.fit(X, y)
        return clf.predict(xTest)

# Random Forest
class modelo3:
    @classmethod
    def predict(cls, xTest):
        fileContent = p.read_csv(fileName)
        X = fileContent.drop('Failure',axis='columns')
        y = fileContent['Failure']
        clf = RandomForestClassifier(n_estimators=500)
        clf.fit(X,y)
        return clf.predict(xTest)

# Extra Tree
class modelo4:
    @classmethod
    def predict(cls, xTest):
        fileContent = p.read_csv(fileName)
        X = fileContent.drop('Failure',axis='columns')
        y = fileContent['Failure']
        clf = ExtraTreeClassifier()
        clf.fit(X,y)
        return clf.predict(xTest)

# Gradient Boosting
class modelo5:
    @classmethod
    def predict(cls, xTest):
        fileContent = p.read_csv(fileName)
        X = fileContent.drop('Failure',axis='columns')
        y = fileContent['Failure']
        clf = GradientBoostingClassifier()
        clf.fit(X,y)
        return clf.predict(xTest)

# Bagging
class modelo6:
    @classmethod
    def predict(cls, xTest):
        fileContent = p.read_csv(fileName)
        X = fileContent.drop('Failure',axis='columns')
        y = fileContent['Failure']
        clf = BaggingClassifier()
        clf.fit(X,y)
        return clf.predict(xTest)