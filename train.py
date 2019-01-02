"""
train.py
作用：将model.py生成的模型包装成trainop训练器，实现误差反向传播和参数更新
成员变量：模型对象model，损失函数cost_fun，损失函数关于输出层的导数cost_fun_der
       学习率eta，训练样本数train_num，正则项系数regu_lambda
成员函数：训练器及参数初始化__init__，单样本的误差反向传播back_propagation
       利用mini_batch的梯度更新参数update_parameters
建模方法：trainop = train(mlpmodel,cost_fun,cost_fun_der,eta,len(training_data),regu_lambda)          
"""
import numpy as np
from activationfun import *

class train():
    
    def __init__(self,model,cost_fun,cost_fun_der,eta,train_num,regu_lambda):
        """
        model是所设计的神经网络
        cost_fun是损失函数
        cost_fun_der是损失函数对输出层的偏导数
        eta是学习率
        """
        self.model = model
        self.cost_fun = cost_fun
        self.cost_fun_der = cost_fun_der
        self.eta = eta
        self.train_num = train_num
        self.regu_lambda = regu_lambda
    
    def back_propagation(self,x,y):
        """
        误差反向传播函数，x和y均为一组样本
        使用反向传播的四个基本方程
        返回损失函数关于连接权值和偏置的梯度
        x,z,a均为行向量(1*m)
        """
        #记录每层各神经元关于连接权值和偏置的梯度，初始化为0
        wgrad_list = [np.zeros(w.shape) for w in self.model.weights]
        bgrad_list = [np.zeros(b.shape) for b in self.model.biases]
        deta_list = [np.zeros(b.shape) for b in self.model.biases]
        #前向传播，计算输入x每层的带权输出z，和激活之后的输出a
        z_list = []; a_list = []
        a_list.append(x); a = x
        for b,w in zip(self.model.biases,self.model.weights):
            z = np.dot(a,w)+b
            z_list.append(z)
            a = sigmoid(z)
            a_list.append(a)
        #利用基本方程1计算输出层神经元的误差detaL，1*n
        detaL = self.cost_fun_der(a_list[-1],y) * \
            sigmoid_derivative(z_list[-1])
        deta_list[-1] = detaL
        #利用基本方程4计算连接权值的梯度，k*n
        wgrad_list[-1] = np.dot(a_list[-2].transpose(),detaL)
        #利用基本方程3计算偏置的梯度，1*n
        bgrad_list[-1] = detaL
        #利用基本方程2将误差进行反向传播，计算各层神经元的误差
        for l in range(2,self.model.layer_num):
            #从倒数第2层开始
            z = z_list[-l]
            #((k*n) * (1*n)')'
            deta_list[-l] = np.dot(self.model.weights[-l+1],deta_list[-l+1].transpose())\
                .transpose() * sigmoid_derivative(z)
            wgrad_list[-l] = np.dot(a_list[-l-1].transpose(),deta_list[-l])
            bgrad_list[-l] = deta_list[-l]
        return {'wgrad':wgrad_list,'bgrad':bgrad_list}
    
    def update_parameters(self,mini_batch):
        """
        计算mini_batch对应的梯度和
        利用梯度和来更新连接权值和偏置
        """
        #将连接权值和偏置均初始化为0
        wgrad_list = [np.zeros(w.shape) for w in self.model.weights]
        bgrad_list = [np.zeros(b.shape) for b in self.model.biases]
        #针对每个训练样本对梯度进行求和
        for x,y in mini_batch:
            x = x.reshape((1,-1))
            y = y.reshape((1,-1))
            deta_grad_dic = self.back_propagation(x,y)
            deta_wgrad_list = deta_grad_dic['wgrad']
            deta_bgrad_list = deta_grad_dic['bgrad']
            wgrad_list = [wg+dewg for wg,dewg in zip(wgrad_list,deta_wgrad_list)]
            bgrad_list = [bg+debg for bg,debg in zip(bgrad_list,deta_bgrad_list)]
        #对参数进行更新
        self.model.weights = [(1-self.eta*(self.regu_lambda/self.train_num))*w-
                          (self.eta/len(mini_batch))*wgrad for w, wgrad in zip(self.model.weights,wgrad_list)]
        self.model.biases = [b-(self.eta/len(mini_batch))*bgrad for b, bgrad in zip(self.model.biases,bgrad_list)]
        
    
    