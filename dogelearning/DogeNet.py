from dogelearning.DogeLayer import *
from dogelearning.DogeOptimizer import *
class Net:


    layers=[]
    batch_size=0
    input_num=0

    def __init__(self,batch_size,input_num):
        self.batch_size=batch_size
        self.input_num=input_num
        pass
    def add(self,layer_type,node_num,activation=""):
        if (len(self.layers)==0):
            last_node_num=self.input_num
        else:
            last_node_num=self.layers[-1].node_num #获取上一层的节点个数
        if (layer_type=='Softmax'):
            self.layers.append(Softmax((node_num)))
        else:
            self.layers.append(Layer(last_node_num,node_num,self.batch_size,activation))
    def forward(self,data):
        for layer in self.layers:
            data=layer.forward(data)
        return data #返回最后输出的data用于反向传播
    def backward(self,y_hat):
        dydx=y_hat
        for layer in reversed(self.layers):
            dydx=layer.backward(dydx)
        return dydx
    def print(self):
        print("网络名                                                      节点个数 激活函数")
        for layer in self.layers:
            print(layer,layer.node_num,layer.activation)

    def bindOptimizer(self,optimizer_name):
        if (optimizer_name=="sgd"):
            for layer in self.layers:
                layer.bindOptimizer(SGDOptimizer(learning_rate=1))
        if (optimizer_name=="momentum"):
            list_optimizer = []
            for layer in self.layers:
                layer.bindOptimizer(MomentumOptimizer(layer.param_names, layer.param_shapes, learning_rate=1))


