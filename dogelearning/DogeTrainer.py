from dogelearning.DogeLayer import Layer
from dogelearning.DogeNet import Net
from tqdm.std import trange
import numpy as np
# import minpy.numpy as np
class Trainer:


    def train(net,train_loader,batch_size,optimizer='sgd',epoch_num=50):

        list = []

        for i in trange(0, epoch_num):
            for batch_idx, (data, target) in enumerate(train_loader):

                if (data.shape[0] < batch_size):
                    break
                data = np.squeeze(data.numpy()).reshape(batch_size, 784)  # 把张量中维度为1的维度去掉,并且改变维度为(64,784)

                target = target.numpy()  # x矩阵 (64,10)
                y_hat = net.forward(data)
                grad=net.backward( np.eye(10)[target] )


                if (batch_idx == 1):
                    list.append(Accuracy(target, y_hat))
                    acc = -np.mean(np.log(y_hat)*np.eye(10)[target])
                    print("loss为"+str(acc))
                    print("准确率为" + str(Accuracy(target, y_hat)))
                    print("梯度均值为"+str(np.mean(grad)))

        return list

def Accuracy(target, y_hat):
        # y_hat.argmax(axis=1)==target 用于比较y_hat与target的每个元素，返回一个布尔数组

        acc = y_hat.argmax(axis=1) == target

        acc = acc + 0  # 将布尔数组转为0，1数组
        return np.mean(acc)  # 通过求均值算出准确率
