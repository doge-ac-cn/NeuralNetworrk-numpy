from torchvision import datasets, transforms
import torch.utils.data as Data
import numpy as np
from tqdm._tqdm import trange
batch_size = 64
learning_rate=0.1
lambd = 3 #正则化惩罚系数
def load_data():


    # 加载torchvision包内内置的MNIST数据集 这里涉及到transform:将图片转化成torchtensor
    train_dataset = datasets.MNIST(root='./data/', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = datasets.MNIST(root='./data/', train=False, transform=transforms.ToTensor())

    # 加载小批次数据，即将MNIST数据集中的data分成每组batch_size的小块，shuffle指定是否随机读取
    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = Data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader,test_loader

def train(train_loader):
    list=[]
    w=np.random.normal(scale=0.01,size=(784,10)) #生成随机正太分布的w矩阵
    b=np.zeros((batch_size,10)) #生成全是0的b矩阵
    for i in trange(0,100):
        for batch_idx, (data, target) in enumerate(train_loader):

            if(data.shape[0]<batch_size):
                break
            data= np.squeeze(data.numpy()).reshape(batch_size,784) # 把张量中维度为1的维度去掉,并且改变维度为(64,784)

            target = target.numpy() #x矩阵 (64,784)

            y_hat=softmax(np.dot(data,w)+b)#预测结果y_hat
            # print(sum(y_hat[0]))
            w_=GradientOfCrossEntropyLoss(data,target,y_hat,type="weight")#计算交叉熵损失函数对于w的梯度
            b_=GradientOfCrossEntropyLoss(data,target,y_hat,type="bias")#计算交叉熵损失函数对于b的梯度
            w=GradientDescent(w,w_+GradientOfL2(w))  #梯度下降优化参数w，权重需要正则化
            b=GradientDescent(b,b_)  #梯度下降优化参数b,偏置不需要正则化
            list.append(Accuracy(target, y_hat))
            if (batch_idx == 50):
                print("准确率为"+str(Accuracy(target, y_hat)))
                print("w的均值为"+str(w.mean()))
    return list

def softmax(label):

    label = np.exp(label.T)#先把每个元素都进行exp运算
    # print(label)
    sum = label.sum(axis=0) #对于每一行进行求和操作

    #print((label/sum).T.sum(axis=1))
    return (label/sum).T #通过广播机制，使每行分别除以各种的和

#梯度下降
def GradientDescent(Param,GradientOfLoss):
    Param = Param -GradientOfLoss/ batch_size * learning_rate
    # print(w_)
    return Param

#计算交叉熵损失函数对于w或者b的梯度
def GradientOfCrossEntropyLoss(data,y,y_hat,type="weight"):
    y = np.eye(10)[y]  # 改为one-hot形式
    data=data.T #(784,64)
    #y_hat-target为预测值与实际值的one-hot挨个做差，得出一个(64,10)的y_hat-y的矩阵

    if (type == "bias"):
        return  y_hat-y
    else:
        return np.dot(data,y_hat-y)

def GradientOfL2(matrix):
    return lambd*matrix
#正则化项是把每个参数平方和，而每个参数对正则化项求偏导都等于它自身

def Accuracy(target,y_hat):
    #y_hat.argmax(axis=1)==target 用于比较y_hat与target的每个元素，返回一个布尔数组
    acc=y_hat.argmax(axis=1) == target
    acc=acc+0  #将布尔数组转为0，1数组
    return acc.mean() #通过求均值算出准确率

if __name__ == '__main__':

    train_loader,test_loader=load_data()    #加载数据(这里使用pytorch加载数据，后面用numpy手写)
    list=train(train_loader)
    import matplotlib.pyplot as plt

    plt.plot(list)
    plt.show()