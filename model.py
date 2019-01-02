"""
model.py
作用：包含model类，生成一个全连接神经网络对象
成员变量：网络层数layer_num，网络维度sizes，连接权重weights，偏置biases
成员函数：网络参数初始化__init__，前向传播forward_propagation，模型保存save_model
建模方法：mlpmodel = model([4,3,2,1])，生成4层，分别具有4，3，2，1个神经元的全连接神经网络
"""
import numpy as np
from activationfun import *

class model():
    
    def __init__(self,sizes):
        """
        sizes是一个列表，每个元素代表一层网络的神经元个数
        """
        #MLP网络的层数，sizes = [4,3,2,1]
        self.layer_num = len(sizes)
        self.sizes = sizes
        #MLP网络每层神经元使用的偏置，输入层没有偏置,偏置定义为行向量
        self.biases = [np.random.randn(1,y) for y in sizes[1:]]
        #MLP网络每层间的连接权值，将神经元错开，初始化网络层之间的权重,[4,3,2]<->[3,2,1]
        #每层之间的连接权值服从高斯分布,均值为0，标准差为1/sqrt(Nin),Nin为输入神经元个数，可以防止神经元饱和
        self.weights = [np.random.randn(y,x)/np.sqrt(y) for y,x in zip(sizes[:-1],sizes[1:])]
            
    def forward_propagation(self,a):
        """
        前向传播函数，a为上一层网络的输出
        x,a均为行向量(1*m)
        """
        for b,w in zip(self.biases,self.weights):
            #a的维度为[1*m],w的维度为[m*n]，a的维度为[1*n]
            a = sigmoid(np.dot(a,w)+b)
        return a
    
    def save_model(self,file_name):
        """
        将模型的维度，连接权值和偏置以字典的形式保存在数据文件中，且为字符串形式
        """
        model_paras_dic = {'sizes':self.sizes,
                     "weights":[w.tolist() for w in self.weights],
                     "biases":[b.tolist() for b in self.biases]}
        f = open(file_name,'w')
        f.write(str(model_paras_dic))
        f.close()
        
    
            
        
        
            
        
        