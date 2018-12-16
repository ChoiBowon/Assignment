# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from sklearn.linear_model import perceptron
from sklearn.metrics import f1_score
import pandas as pd
import numpy as np

tr = pd.read_csv("mnist_train.csv")
ts = pd.read_csv("mnist_test.csv")
tr = np.array(tr)
ts = np.array(ts)
tr_y, tr_x = np.split(tr,[1], axis = 1)
ts_y, ts_x = np.split(ts,[1], axis = 1)

def random_X(p):
    random = np.random.normal(0, 1, (2**p, 784))
    return random

def A(X, random):
    return np.dot(random, X.T).T

for i in range(15):
    r_arr = random_X(i)
    tr_A = A(tr_x, r_arr)
    ts_A = A(ts_y, r_arr)
    pct = perceptron()
    pct.fit(tr_A, tr_y)
    y_pred = pct.predict(ts_A)
    print(f1_score(ts_y, y_pred))