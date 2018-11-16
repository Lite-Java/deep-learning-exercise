# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 18:43:00 2018

@author: liuhuan
"""
import numpy as np
class Perceptron:
    def __init__(self, N, alpha=0.02):
        #self.W = np.random.randn(N + 1) / np.sqrt(N)
        
        self.W=np.array([0.5,0.5,0.75])
        self.alpha = alpha 

    def sign(self, x):
        return 1 if x > 0 else -1 

    def fit(self, X, y, epochs=10):
        X = np.c_[X, np.ones((X.shape[0]))]

        for epoch in np.arange(0, epochs):
            for (x, target) in zip(X, y):
                p = self.sign(np.dot(x, self.W))
                print("W:{}".format(self.W))
                if p != target:
                    error = p - target
                    self.W += -self.alpha * error * x
                    

    def predict(self, X, addBias=True):
        X = np.atleast_2d(X)

        if addBias:
            X = np.c_[X, np.ones((X.shape[0]))]

        return self.step(np.dot(X, self.W))



X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_or = np.array([[0], [1], [1], [1]])
y_and = np.array([[-1], [-1], [-1], [1]])
y_xor = np.array([[-1], [1], [1], [-1]])

#print("[INFO] training perceptron....")
#p = Perceptron(X.shape[1], alpha=0.1)
#p.fit(X, y_or, epochs=20)

#print("[INFO] testing perceptron OR...")
#for (x, target) in zip(X, y_or):
#    pred = p.predict(x)
#    print("[INFO] data={}, ground_truth={}, pred={}".format(x, target[0], pred))

#print("[INFO] training perceptron AND....")
#p = Perceptron(X.shape[1], alpha=0.02)
#p.fit(X, y_and, epochs=100)

#print("[INFO] testing perceptron AND...")
#for (x, target) in zip(X, y_and):
#    pred = p.predict(x)
#    print("[INFO] data={}, ground_truth={}, pred={}".format(x, target[0], pred))
#
print("[INFO] training perceptron XOR....")
p = Perceptron(X.shape[1], alpha=0.1)
p.fit(X, y_xor, epochs=20)
#
#print("[INFO] testing perceptron XOR...")
#for (x, target) in zip(X, y_xor):
#    pred = p.predict(x)
#    print("[INFO] data={}, ground_truth={}, pred={}".format(x, target[0], pred))
#print("X.shape\n", X.shape)
#print("X.shape[0]\n", X.shape[0])
#print("X.shape[1]\n", X.shape[1])

