"""
activationfun.py
作用：激活函数sigmoid，激活函数的导数sigmoid_derivative
    softmax函数softmax
"""
import numpy as np
    
def sigmoid(x):
    """
    sigmoid激活函数
    """
    return 1.0/(1.0+np.exp(-x))
    
def sigmoid_derivative(x):
    """
    sigmoid函数的导数
    """
    return sigmoid(x)*(1-sigmoid(x))

def softmax(x):
    """
    softmax函数，返回softmax后的向量
    """
    exps = np.exp(x-np.max(x))
    return exps/np.sum(exps)