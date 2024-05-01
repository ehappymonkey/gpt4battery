import torch
import numpy as np


def gradient_vector_function(f, x):
    n = len(x)
    m = len(f(x))
    gradient = np.zeros((m, n))
    
    h = 1e-6  # 微小增量
    
    for i in range(n):
        x_plus_h = np.copy(x)
        x_plus_h[i] += h
        gradient[:, i] = (f(x_plus_h) - f(x)) / h
    
    return gradient


def physical_con(u, i, t, theta1, theta2):
    obj = theta1*i + theta2*u + gradient_vector_function(u, u)
    return obj