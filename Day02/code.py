import os
import time
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import paddle
import paddle.fluid as fluid
import paddle.fluid.layers as layers
from multiprocessing import cpu_count
from paddle.fluid.dygraph import Pool2D,Conv2D
from paddle.fluid.dygraph import Linear

def data_mapper(sample):
    img, label = sample
    img = Image.open(img)
    img = img.resize((100, 100), Image.ANTIALIAS)
    img = np.array(img).astype('float32')
    img = img.transpose((2, 0, 1))
    img = img/255.0
    return img, label

def data_reader(data_list_path):
    def reader():
        with open(data_list_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                img, label = line.split('\t')
                yield img, int(label)
    return paddle.reader.xmap_readers(data_mapper, reader, cpu_count(), 512)

# 用于训练的数据提供器
train_reader = paddle.batch(reader=paddle.reader.shuffle(reader=data_reader('/home/aistudio/train_data.list'), buf_size=100), batch_size=128)
# 用于测试的数据提供器
test_reader = paddle.batch(reader=data_reader('/home/aistudio/test_data.list'), batch_size=1) 

class ResNet(fluid.dygraph.Layer):
    def __init__(self, layers=50, class_dim=10):
        super(ResNet, self).__init__()

        self.layers = layers
        supported_layers = [50, 101, 152]
        assert layers in supported_layers, \
            "supported layers are {} but input layer is {}".format(supported_layers, layers)

        if layers == 50:
            depth = [3, 4, 6, 3]
        elif layers == 101:
            depth = [3, 4, 23, 3]
        elif layers == 152:
            depth = [3, 8, 36, 3]
        num_channels = [64, 256, 512, 1024]
        num_filters = [64, 128, 256, 512]

        self.conv = ConvBNLayer(
            num_channels=3,
            num_filters=64,
            filter_size=7,
            stride=2,
            act='relu')
        self.pool2d_max = Pool2D(
            pool_size=3,
            pool_stride=2,
            pool_padding=1,
            pool_type='max')

        self.bottleneck_block_list = []
        for block in range(len(depth)):
            shortcut = False
            for i in range(depth[block]):
                bottleneck_block = self.add_sublayer(
                    'bb_%d_%d' % (block, i),
                    BottleneckBlock(
                        num_channels=num_channels[block]
                        if i == 0 else num_filters[block] * 4,
                        num_filters=num_filters[block],
                        stride=2 if i == 0 and block != 0 else 1,
                        shortcut=shortcut))
                self.bottleneck_block_list.append(bottleneck_block)
                shortcut = True

        self.pool2d_avg = Pool2D(
            pool_size=7, pool_type='avg', global_pooling=True)

        self.pool2d_avg_output = num_filters[len(num_filters) - 1] * 4 * 1 * 1

        import math
        stdv = 1.0 / math.sqrt(2048 * 1.0)

        self.out = Linear(self.pool2d_avg_output,
                      class_dim,
                      act='softmax',
                      param_attr=fluid.param_attr.ParamAttr(
                          initializer=fluid.initializer.Uniform(-stdv, stdv)))

    def forward(self, inputs):
        y = self.conv(inputs)
        y = self.pool2d_max(y)
        for bottleneck_block in self.bottleneck_block_list:
            y = bottleneck_block(y)
        y = self.pool2d_avg(y)
        y = fluid.layers.reshape(y, shape=[-1, self.pool2d_avg_output])
        y = self.out(y)
        return y

##################################################################################
#用动态图进行训练
with fluid.dygraph.guard():
    #model=MyDNN() #模型实例化
    model=ResNet(class_dim=10)
    model.train() #训练模式
    # opt=fluid.optimizer.SGDOptimizer(learning_rate=0.01, parameter_list=model.parameters())#优化器选用SGD随机梯度下降，学习率为0.001.
    opt=fluid.optimizer.Adam(learning_rate=0.0001, parameter_list=model.parameters())
    epochs_num=30 #迭代次数
    
    for pass_num in range(epochs_num):
        
        for batch_id,data in enumerate(train_reader()):
            
            images=np.array([x[0].reshape(3,100,100) for x in data],np.float32)
            # print(images.shape)
            labels = np.array([x[1] for x in data]).astype('int64')
            labels = labels[:, np.newaxis]
            image=fluid.dygraph.to_variable(images)
            label=fluid.dygraph.to_variable(labels)
            label.stop_gradient = True
            predict=model(image)#预测
            # print(predict)
            loss=fluid.layers.cross_entropy(predict,label)
            avg_loss=fluid.layers.mean(loss)#获取loss值
            
            acc=fluid.layers.accuracy(predict,label)#计算精度
            
            if batch_id!=0 and batch_id%1==0:
                print("train_pass:{},batch_id:{},train_loss:{},train_acc:{}".format(pass_num,batch_id,avg_loss.numpy(),acc.numpy()))
            
            avg_loss.backward()
            opt.minimize(avg_loss)
            model.clear_gradients()
            
    fluid.save_dygraph(model.state_dict(),'resnet50')#保存模型


##################################################################################
#模型校验
with fluid.dygraph.guard():
    accs = []
    model_dict, _ = fluid.load_dygraph('resnet50')
    #model = MyDNN()
    model=ResNet(class_dim=10)
    model.load_dict(model_dict) #加载模型参数
    model.eval() #训练模式
    for batch_id,data in enumerate(test_reader()):#测试集
        images=np.array([x[0].reshape(3,100,100) for x in data],np.float32)
        
        labels = np.array([x[1] for x in data]).astype('int64')
        labels = labels[:, np.newaxis]

        image=fluid.dygraph.to_variable(images)
        label=fluid.dygraph.to_variable(labels)
        
        predict=model(image)       
        acc=fluid.layers.accuracy(predict,label)
        accs.append(acc.numpy()[0])
    avg_acc = np.mean(accs)
    print(avg_acc)

#读取预测图像，进行预测

def load_image(path):
    img = Image.open(path)
    img = img.resize((100, 100), Image.ANTIALIAS)
    img = np.array(img).astype('float32')
    img = img.transpose((2, 0, 1))
    img = img/255.0
    # print(img.shape)
    return img

##################################################################################
# #构建预测动态图过程
with fluid.dygraph.guard():
    infer_path = '/home/aistudio/work/DNN/手势.JPG'
    #model=MyDNN()#模型实例化
    model=ResNet(class_dim=10)
    model_dict,_=fluid.load_dygraph('resnet50')
    model.load_dict(model_dict)#加载模型参数
    model.eval()#评估模式
    infer_img = load_image(infer_path)
    infer_img=np.array(infer_img).astype('float32')
    infer_img=infer_img[np.newaxis,:, : ,:]
    infer_img = fluid.dygraph.to_variable(infer_img)
    result=model(infer_img)
    print("#"*100)
    print("the predict number of '手势.JPG' is : ",np.argmax(result.numpy()))
    print("#"*100)
