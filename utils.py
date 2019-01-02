"""
utils.py
作用：读取训练集和测试集中的图片，将其转为行向量，并用列表存储
     特征向量与标签的元组，并返回
"""
import os
import PIL.Image
import numpy as np
   
def load_pic_feature(pic_dir):
    """
    读入一张图片，先将其大小设置为28*28
    再转为灰度图片，最后将其像素值重塑为
    1*784的行向量
    """
    im = PIL.Image.open(pic_dir)
    #图片设置为28*28
    x_s = 28
    y_s = 28
    out = im.resize((x_s, y_s), PIL.Image.ANTIALIAS) 
    #转成灰度图像
    im_arr = np.array(out.convert('L'))
    num0 = 0
    num255 = 0
    threshold = 100
    for x in range(x_s):
        for y in range(y_s):
            if im_arr[x][y] > threshold : num255 = num255 + 1
            else : num0 = num0 + 1
    if(num255 > num0) :
        print("convert!")
        for x in range(x_s):
            for y in range(y_s):
                im_arr[x][y] = 255 - im_arr[x][y]
                if(im_arr[x][y] < threshold) :  im_arr[x][y] = 0
    out = PIL.Image.fromarray(np.uint8(im_arr))
    #将像素点重塑维1*784维的行向量
    nm = im_arr.reshape((1, 784))
    nm = nm.astype(np.float32)
    nm = np.multiply(nm, 1.0 / 255.0)
    return nm

def label_one_hot(label_id,n_classes):
    """
    将标签y进行one-hot编码
    """
    y = [0] * n_classes
    y[int(label_id)] = 1.0
    return np.array(y)

def load_train_feature(pic_dir):
    """
    输入存储训练图片的路径，按类别文件夹读取
    每张图片，并调用load_pic_feature函数，将
    转化后的特征与标签的one-hot向量组成元组
    再用一个列表存储该样本对应的元组
    """
    training_data = []
    path = pic_dir
    dir_list = os.listdir(path)
    for dir_name in dir_list:
        y = label_one_hot(int(dir_name),10)
        file_list = os.listdir(os.path.join(path,dir_name))
        for file_name in file_list:
            x = load_pic_feature(os.path.join(path,os.path.join(dir_name,file_name)))
            training_data.append((x,y))
    return training_data

def load_test_feature(pic_dir):
    """
    输入存储测试图片的路径，按类别文件夹读取
    每张图片，并调用load_pic_feature函数，将
    转化后的特征与原标签直接组成元组
    再用一个列表存储该样本对应的元组
    """
    test_data = []
    path = pic_dir
    dir_list = os.listdir(path)
    for dir_name in dir_list:
        y = int(dir_name)
        file_list = os.listdir(os.path.join(path,dir_name))
        for file_name in file_list:
            x = load_pic_feature(os.path.join(path,os.path.join(dir_name,file_name)))
            test_data.append((x,y))
    return test_data