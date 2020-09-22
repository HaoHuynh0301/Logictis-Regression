import pandas as pd
import numpy as np
import matplotlib as mpl
import sklearn as sk
import math

ITERS=100000
LEARNING_RATE=0.01

dataset=pd.read_csv('data_classification.csv', header = None)
N, d = dataset.shape
data_value=dataset.values[:, 0:d - 1].reshape(-1, d - 1)
data_target=dataset.values[:, 2].reshape(-1, 1)

def sigmoid(z):
    return 1/(1+np.exp(-z))

def distinct(p):
    if p>=0.5:
        return 1
    return 0

def predict(data_value, weights):
    z=np.dot(data_value, weights)
    return sigmoid(z)

def update_weight(data_value, data_target, weights, learning_rate):
    n=len(data_value)
    prediction=predict(data_value, weights)
    gd=np.dot(data_value.T, (prediction-data_target))
    gd=gd/n*learning_rate-gd
    weights=weights-gd
    return weights

def train(data_value, data_target, weights, learning_rate, iters):
    n=len(data_value)
    for i in range(n):
        weights=update_weight(data_value, data_target, weights, learning_rate)

    return weights

data_value = np.hstack((np.ones((N, 1)), data_value))
weights = np.array([0., 0.1, 0.1]).reshape(-1, 1)

weights=train(data_value, data_target, weights, LEARNING_RATE, ITERS)

predict_data=np.array([[4.534543, 7,3123123]])

predict_result=predict(predict_data, weights)

print(predict_result)

# print(dataset.values)






