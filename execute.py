"""
execute.py
作用：训练mlp网络，并将模型保存
"""
import numpy as np
from model import *
from train import *
from utils import *
from activationfun import *
import random
import time
import sys

def cost_fun(a,y):
    #使用softmax+交叉熵作为损失函数
    a_s = -np.log(softmax(a))
    return np.sum(np.nan_to_num(a_s*y))

def cost_fun_der(a,y):
    #将loss对输出层神经元求导
    a_s = softmax(a)
    index = np.argmax(y)
    a_s[0,index] = a_s[0,index]-1
    return a_s

def training_avg_loss(model,training_data,regu_lambda):
    #返回对全体训练集上的平均损失
    loss = 0.0
    for x, y in training_data:
        x = x.reshape((1,-1))
        y = y.reshape((1,-1))
        a = model.forward_propagation(x)
        loss += cost_fun(a,y)
    loss = loss/len(training_data)
    loss += (0.5*(regu_lambda/len(training_data))*sum(
            np.linalg.norm(w)**2 for w in model.weights))
    return loss

def testing_accuracy(model,test_data):
    #返回对测试集上的预测精度
    test_results = [(np.argmax(softmax(model.forward_propagation(x.reshape((1,-1))))), y)
                       for (x, y) in test_data]
    test_accury = sum(int(x == y) for (x, y) in test_results)/len(test_data)
    return test_accury

if __name__ == '__main__':
    #加载手写数字训练图片
    train_pic_dir = sys.argv[1] #'./mnist_pic/mnist_train/' 
    training_data = load_train_feature(train_pic_dir)
    print("训练样本数:",len(training_data))
    #加载手写数字测试图片
    test_pic_dir = sys.argv[2] #'./mnist_pic/mnist_test/' 
    test_data = load_test_feature(test_pic_dir)
    print("测试样本数:",len(test_data))
    #初始化网络的超参数
    eta = 0.5
    regu_lambda = 0.05
    epoch_num = 300
    batch_size = 10
    #生成MLP网络结构并初始化参数
    mlpmodel = model([784,500,10])
    #生成网络的训练器
    trainop = train(mlpmodel,cost_fun,cost_fun_der,eta,len(training_data),regu_lambda)
    #开始训练
    start = time.time()
    for i in range(epoch_num):
        random.shuffle(training_data)
        mini_batches = [training_data[k:k+batch_size] for k in 
                        range(0,len(training_data),batch_size)]
        for mini_batch in mini_batches:
            trainop.update_parameters(mini_batch)
        #每个epoch输出一次全体训练集的loss
        train_avg_loss = training_avg_loss(mlpmodel,training_data,regu_lambda)
        #每个epoch输出一次测试集的预测精度
        test_accuracy = testing_accuracy(mlpmodel,test_data)
        print("epoch_num:{0},train_loss:{1},test_accuracy:{2}".format(i,train_avg_loss,test_accuracy))
    end = time.time()
    print("训练时间:",end-start)
    #模型保存
    file_name = sys.argv[3] #'mlpmodel.data'
    mlpmodel.save_model(file_name)