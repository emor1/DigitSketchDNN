"""
0から作るディープラーニング4章を参考に実装

The MIT License (MIT)
Copyright (c) 2016 Koki Saitoh
https://github.com/oreilly-japan/deep-learning-from-scratch/blob/master/ch04/gradient_simplenet.py
"""


import numpy as np
from functions.function import cross_entropy_error, numerical_gradient, softmax

class SimpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss