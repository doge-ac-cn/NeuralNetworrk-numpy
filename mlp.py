from torchvision import datasets, transforms
import torch.utils.data as Data
# import minpy.numpy as mnp
# import numpy as np

from dogelearning.DogeNet import Net
from dogelearning.DogeLayer import Layer
from dogelearning.DogeTrainer import Trainer

batch_size = 256
learning_rate=0.001

def load_data():
    # 加载torchvision包内内置的MNIST数据集 这里涉及到transform:将图片转化成torchtensor
    train_dataset = datasets.MNIST(root='./data/', train=True, transform=transforms.ToTensor(), download=True)
    # 加载小批次数据，即将MNIST数据集中的data分成每组batch_size的小块，shuffle指定是否随机读取
    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = datasets.MNIST(root='./data/', train=False, transform=transforms.ToTensor())
    test_loader = Data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader,test_loader

if __name__ == '__main__':

    train_loader,test_loader=load_data()    #加载数据(这里使用pytorch加载数据，后面用numpy手写)
    net = Net(batch_size,784)
    print(net.batch_size)
    # net.add("", 100, activation="Sigmoid")
    # net.add("", 100, activation="Sigmoid")
    # net.add("", 100, activation="Sigmoid")
    # net.add("", 100, activation="Sigmoid")
    # net.add("", 10, activation="Sigmoid")
    # net.add("Softmax", 10)
    net.add("", 100, activation="Tahn")
    net.add("", 100, activation="Tahn")
    net.add("", 100, activation="Tahn")
    net.add("", 100, activation="Tahn")
    net.add("", 10, activation="Tahn")
    net.add("Softmax", 10)
    # net.add("", 100, activation="Relu")
    # net.add("", 100, activation="Relu")
    # net.add("", 100, activation="Relu")
    # net.add("", 100, activation="Relu")
    # net.add("", 10, activation="Relu")
    # net.add("Softmax", 10)
    net.print()
    # net.bindOptimizer("sgd")
    net.bindOptimizer("momentum")
    list=Trainer.train(net,train_loader,batch_size,epoch_num=200)

    import matplotlib.pyplot as plt
    plt.plot(list)
    plt.show()

    list = Trainer.train(net, test_loader, 256, epoch_num=2)
    print(list)