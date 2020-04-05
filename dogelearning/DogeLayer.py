import numpy as np
# import minpy.numpy as np
class Layer:
    lamda = 0.1  # 正则化惩罚系数
    param_shapes=[]
    activation = ''
    def __init__(self,last_node_num,node_num,batch_size,activation):
        self.last_node_num=last_node_num
        self.node_num=node_num
        self.w = np.random.normal(scale=0.1, size=(last_node_num,node_num))  # 生成随机正太分布的w矩阵
        self.b = np.random.normal(scale=0.1, size=(batch_size, node_num))
        self.activation=activation
        self.batch_size=batch_size
        self.param_names=['w','b']
        self.param_shapes=[(last_node_num,node_num),(batch_size, node_num)]
    def forward(self,data):
        self.input=data
        data=np.dot(data, self.w) + self.b
        if self.activation=="Sigmoid":
            data=1 / (1 + np.exp(-data))
            # print(data.mean())
        if self.activation=="Tahn":
            data = (np.exp(data)- np.exp(-data)) / (np.exp(data)+ np.exp(-data))

        if self.activation == "Relu":
            data= (np.abs(data)+data)/2.0

        self.output = data
        return data
    def backward(self,grad):
        if self.activation == "Sigmoid":
            grad = self.output * (1 - self.output) * grad
        if self.activation == "Tahn":
            grad = (1 - self.output**2) * grad
        if self.activation=="Relu":
            self.output[self.output <= 0] = 0
            self.output[self.output > 0] = 1
            grad = self.output *grad
        w_grad=(np.dot(self.input.T, grad) + (self.lamda * self.w))/self.batch_size
        b_grad=grad/self.batch_size
        grad=np.dot(grad,self.w.T)
        self.w = self.optimizer.update('w',self.w, w_grad )
        self.b =self.optimizer.update('b',self.b,b_grad )
        # self.w = self.w - (w_gradient+(self.lamda*self.w)) / self.batch_size * self.learning_rate
        # self.b = self.b - b_gradient / self.batch_size * self.learning_rate
        return grad

    def bindOptimizer(self,optimizer):
        self.optimizer=optimizer


class Softmax (Layer):
    y_hat=[]
    def __init__(self,node_num):
        self.param_names=[]
        self.node_num=node_num
        pass
    def forward(self,data):
        data = np.exp(data.T)  # 先把每个元素都进行exp运算
        # print(label)
        sum = np.sum(data,axis=0)  # 对于每一行进行求和操作
        # print((label/sum).T.sum(axis=1))
        self.y_hat=(data / sum).T
        return self.y_hat  # 通过广播机制，使每行分别除以各种的和
    def backward(self,y):
        return self.y_hat-y