# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 12:33:13 2020

@author: IKM1YH
"""
import numpy as np
def generate_noisy_points(n=10, noise_variance=1e-6):
    np.random.seed(777)
    X = np.random.uniform(-3., 3., (n, 1))
    y = np.sin(X) + np.random.randn(n, 1) * noise_variance**0.5
    return X, y

import matplotlib.pylab as plt
X, y = generate_noisy_points()
plt.plot(X, y, 'x')
plt.show()

Xtest, ytest = generate_noisy_points(100)
Xtest.sort(axis=0)