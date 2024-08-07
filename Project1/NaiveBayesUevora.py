import numpy as n
import pandas as p
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class NaiveBayesUevora:
    alpha = float

    def __init__(self, alpha = 0):
        self.alpha = alpha
        self.columns = list
        self.noProperties = {}
        self.PA = {}
        self.PAB = {}
        self.classes = {}
        self.aTraining = n.array
        self.bTraining = n.array
        self.sizeTraining = int
        self.noColumns = int

    def fit(self, x, y):
        self.columns = list(x.columns) # Classes
        self.aTraining = x # Todas as classes excepto a última
        self.bTraining = y # Apenas a última
        self.sizeTraining = x.shape[0] # Nro de linhas
        self.noColumns = x.shape[1] # Nro de classes
        for attribute in self.columns:
            self.PAB[attribute] = {}
            self.PA[attribute] = {}
            self.noProperties[attribute] = len(n.unique(self.aTraining[attribute]))

            for xValue in n.unique(self.aTraining[attribute]):
                self.PA[attribute][xValue] = 0
                self.PAB[attribute][xValue] = {}
                
                for yValue in n.unique(self.bTraining):
                    self.PAB[attribute][xValue][yValue] = 0
                    self.classes[yValue] = 0

        # P(A)
        for yValue in n.unique(self.bTraining):
            noOcorencias = sum(self.bTraining == yValue)
            self.classes[yValue] = (noOcorencias + self.alpha) / (self.sizeTraining + (self.alpha * len(n.unique(self.bTraining))))

        # P(B|A)
        for attribute in self.columns:
            for yValue in n.unique(self.bTraining):
                noOcorenciasY = sum(self.bTraining == yValue)
                noOcorenciasXY = self.aTraining[attribute][self.bTraining[self.bTraining == yValue].index.values.tolist()].value_counts().to_dict()
                for xyValue, noOcorencias in noOcorenciasXY.items():
                    self.PAB[attribute][xyValue][yValue] = (noOcorencias + self.alpha) / (noOcorenciasY + (self.alpha * self.noProperties[attribute]))

    def predict(self, x):
        results = []
        x = n.array(x)
        for test in x:
            possibleResults = {}
            for yValue in n.unique(self.bTraining):
                PA = self.classes[yValue]
                PBA = 1
                for attribute, property in zip(self.columns, test):
                    if property not in n.unique(self.aTraining[attribute]):
                        self.add_unfound_property(attribute, property)
                    PBA *= self.PAB[attribute][property][yValue]
                possibleResults[yValue] = PBA * PA
            result = max(possibleResults, key=lambda x: possibleResults[x])
            results.append(result)
        return n.array(results)

    def add_unfound_property(self, attribute, property):
        self.PAB[attribute][property] = {}
        for yValue in n.unique(self.bTraining):
            self.PAB[attribute][property][yValue] = 0
            noOcorrenciasY = sum(self.bTraining == yValue)
            noOcorrenciasXY = 0
            self.PAB[attribute][property][yValue] = (noOcorrenciasXY + self.alpha) / (noOcorrenciasY + (self.alpha * self.noProperties[attribute]))

    def accuracy_score(self, x, y):
        prev = self.predict(x)
        return round(float((sum(prev == y)) / float(len(y)) * 100), 2)

    def precision_score(self, x, y):
        results = []
        prev = self.predict(x)
        for yValue in n.unique(self.bTraining):
            vp = 0
            fp = 0
            for value, yPrevValue in zip(y, prev):
                if yPrevValue == yValue and value == yPrevValue:
                    vp += 1
                elif yPrevValue == yValue and value != yPrevValue:
                    fp += 1
            if vp == 0 and fp == 0:
                results.append(0)
            else:
                results.append(vp / (vp + fp))
        return round(float(sum(results)/len(results)*100),2)
