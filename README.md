# annBasedNumpy
>只利用numpy编写多层全连接神经网络，并以mnist数据进行测试，单隐层500个节点的网络下，在mnist提供的测试集上识别精度为98%

* activationfun.py
>sigmoid函数和softmax函数
* model.py
>可设置多层多节点的全连接神经网络
* utils.py
>将图片转为数组
* train.py
>误差反向传播的四个公式及参数梯度下降，并生成trainop优化器
* inference.py
>测试接口，支持读入图片并给出结果
