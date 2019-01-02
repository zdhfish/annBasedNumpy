"""
inference.py
作用：加载模型，对任意图片，任意文件夹中图片或有标签图片集，完成测试，并给出预测精度
"""
import numpy as np
from model import *
from utils import *
from activationfun import *
import sys

def load_model(file_name):
    """
    加载模型参数，并重新赋值给模型对象
    """
    f = open(file_name,'r',encoding='utf-8')
    model_paras_str = f.read()
    model_paras_dic = eval(model_paras_str)
    f.close()
    weights = [np.array(w) for w in model_paras_dic["weights"]]
    biases = [np.array(b) for b in model_paras_dic["biases"]]
    sizes = model_paras_dic['sizes']
    mlpmodel = model(sizes)
    mlpmodel.weights = weights
    mlpmodel.biases = biases
    return mlpmodel

def testing_accury(model,test_pic_dir):
    #返回对测试集上的预测精度
    test_data = load_test_feature(test_pic_dir)
    test_results = [(np.argmax(softmax(model.forward_propagation(x.reshape((1,-1))))), y)
                       for (x, y) in test_data]
    test_accury = sum(int(x == y) for (x, y) in test_results)/len(test_data)
    return test_results,test_accury

def testing_result(model,test_pic_dir):
    #直接返回测试图片结果
    x = load_pic_feature(test_pic_dir)
    return np.argmax(softmax(model.forward_propagation(x.reshape((1,-1)))))

if __name__ == '__main__':
    #加载模型
    file_name = sys.argv[1] #'mlpmodel.data'
    #加载测试图片
    pic_dir = sys.argv[2] #'./mnist_pic/mnist_test/3/mnist_test_1062.png'
    #对一个文件夹中图片预测，将结果保存的路径
    save_dir = sys.argv[3]
    #是否单张测试标志
    flag = sys.argv[4] #1
    mlpmodel = load_model(file_name)
    #进行测试
    if int(flag) == 1:        
        result = testing_result(mlpmodel,pic_dir)
        print("识别结果为：",result)
    elif int(flag) == 0:
        test_results,test_accury = testing_accury(mlpmodel,pic_dir)
        print("测试精度为：",test_accury)
    else:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        file_list = os.listdir(pic_dir)
        with open(save_dir+'/test_result.data','w',encoding='utf-8') as fout:
            for file_name in file_list:
                file_path = os.path.join(pic_dir,file_name)
                result = str(testing_result(mlpmodel,file_path))
                print("file_name:{0},predict_result:{1}".format(file_name,result))
                fout.write("file_name:"+file_name+','+"predict_result:"+result+"\n")
        print("结果保存完成！")
      
            


