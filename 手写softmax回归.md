## softmax诞生原因

线性回归主要用于连续值预测，即回归问题，比如判定一个东西是鸡的概率是多少

而当模型需要预测多个离散值时，即分类问题，比如判定一个东西是鸡还是鹅还是鸭

此时就需要多个输出单元，并且修改运算方式从而方便预测和训练，这也就是softmax层诞生的原因


这里使用的是minist手写数字识别数据集

## 1.首先通过Pytorch读取数据集

```python
def load_data():
    train_dataset = datasets.MNIST(root='./data/',
                                   train=True,
                                    transform=transforms.ToTensor(),
                                   download=True)

    test_dataset = datasets.MNIST(root='./data/',
                                  transform=transforms.ToTensor(),
                                  train=False)

    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)
    return train_loader,test_loader
```

## 2.softmax回归模型

这里每个输入都是$$28*28$$的数据，即$$784*1$$

这里我选择每次都输入batch_size为64的数据进行优化

即输入数据shape为(64,784)

即64*784的矩阵
$$
\left[

 \matrix{  

x_{1,1} & x_{2,1} & ...&x_{783,1} &x_{784,1} \\  
x_{1,2} & x_{2,2} & ...&x_{783,2} &x_{784,2} \\ 
&  & ...& &\\
x_{1,63} & x_{2,63} & ...&x_{783,63} &x_{784,63} \\ 
x_{1,64} & x_{2,64} & ...&x_{783,64} &x_{784,64} \\  } 

\right]
$$


softmax和线性回归一样都是对将输入特征用线性函数计算输出值，但是这里有一点不同，就是softmax回归的输出值个数不再是一个，而是等于类别数，这里输出类别为0123456789十个数字，因此有十个输出值

因此此时需要十个线性函数来对应这十个输入值到输出值的转化
$$
y_1=W_{1,1}*x_1+W_{2,1}*x_2+...+W_{783,1}*x_{783,1}+W_{784,1}*x_{784,1}+b_{1}\\
y_2=W_{1,2}*x_1+W_{2,2}*x_2+...+W_{783,2}*x_{783,2}+W_{784,2}*x_{784,2}+b_{2}\\
...\\
...\\
y_{10}=W_{1,10}*x_1+W_{2,10}*x_2+...+W_{783,10}*x_{783,10}+W_{784,10}*x_{784,10}+b_{10}
$$
其中w矩阵为(784,10),即
$$
\left[

 \matrix{  

W_{1,1} & W_{2,1} & ...&W_{783,1} &W_{784,1} \\  
W_{1,2} & W_{2,2} & ...&W_{783,2} &W_{784,2} \\ 
&  & ...& &\\
W_{1,9} & W_{2,9} & ...&W_{783,9} &W_{784,9} \\ 
W_{1,10} & W_{2,10} & ...&W_{783,10} &W_{784,10} \\  } 

\right]
$$
b矩阵为(64,10)

即
$$
\left[

 \matrix{  

b_{1}&b_{2}&...&b_{9}&b_{10}  \\  
b_{1}&b_{2}&...&b_{9}&b_{10} \\
&&...&&\\
b_{1}&b_{2}&...&b_{9}&b_{10}  \\ 
b_{1}&b_{2}&...&b_{9}&b_{10}   \\  } 

\right]
$$
最终为
$$
y=\left[

 \matrix{  

W_{1,1} & W_{2,1} & ...&W_{783,1} &W_{784,1} \\  
W_{1,2} & W_{2,2} & ...&W_{783,2} &W_{784,2} \\ 
&  & ...& &\\
W_{1,9} & W_{2,9} & ...&W_{783,9} &W_{784,9} \\ 
W_{1,10} & W_{2,10} & ...&W_{783,10} &W_{784,10} \\  } 

\right]
*
\left[

 \matrix{  

x_{1,1} & x_{2,1} & ...&x_{783,1} &x_{784,1} \\  
x_{1,2} & x_{2,2} & ...&x_{783,2} &x_{784,2} \\ 
&  & ...& &\\
x_{1,63} & x_{2,63} & ...&x_{783,63} &x_{784,63} \\ 
x_{1,64} & x_{2,64} & ...&x_{783,64} &x_{784,64} \\  } 

\right]
+\left[

 \matrix{  

b_{1}&b_{2}&...&b_{9}&b_{10}  \\  
b_{1}&b_{2}&...&b_{9}&b_{10} \\
&&...&&\\
b_{1}&b_{2}&...&b_{9}&b_{10}  \\ 
b_{1}&b_{2}&...&b_{9}&b_{10}   \\  } 

\right]
$$
y矩阵最终等于$$x(64,784)*w(784,10)+b(64,10)$$

其中$$x(64,784)*w(784,10)$$结果为$$(64,10)$$,加上b最后还是一个$$(64,10)$$的矩阵

对应一个批次64个数据中中，预测十个数字每个数字的可能性大小

代码如下

```python
def train():
    w=np.random.randn(784,10) #生成随机正太分布的w矩阵
    b=np.zeros((64,10)) #生成全是0的b矩阵
    for batch_idx, (data, target) in enumerate(train_loader):
        data= np.squeeze(data.numpy()).reshape(64,28*28) 
        # 把张量中维度为1的维度去掉,并且改变维度为(64,784)
        target = target.numpy() #x矩阵 (64,784)
        y_hat=np.dot(data,w)+b
        print(y_hat.shape)
```

最终y_hat为$$(64,10)$$

![image][link1]

## 3.softmax运算

假设预测值分别是1,10,...100，此时直接通过输出值判断，$$y_{10}$$最大识别结果为数字9

但是这样会使得，不知道怎么划分这些值到底是很大还是很小，也就是难以去优化参数

比如如果这次里100最大，但是下次中10000最大而且最小的都有1000，那么100到底算不算大

这里就用softmax运算来解决
$$
\hat{y_1},\hat{y_2}...\hat{y_{10}}=softmax(output_1,output_2...output_{10})\\
其中\\
\hat{y_j}=\frac{exp(output_j)}{\sum_{i=1}^{10}{exp(output_i)}}
$$
即每个$$\hat{y_i}$$为exp(当前输出值)/所有exp(输出值)求和

这样一来，最终所有$$\hat{y_i}$$的和为1，而且每个$$\hat{y_i}$$对应的都是对每个手写数字概率的预测

```python
def softmax(label):
    #label时一个(64,10)的numpy数组
    label = np.exp(label.T)#先把每个元素都进行exp运算
    sum = label.sum(axis=0) #对于每一行进行求和操作
    # print((label/sum).T.shape)
    # print((label/sum).T)
    return (label/sum).T #通过广播机制，使每行分别除以各种的和
```

## 4.交叉熵损失函数

这里的损失函数我们可以直接采用平方损失函数，即$$（\hat{y_i}-y)^2$$

但是想要预测结果正确，我们不需要使得预测值和实际值完全一致

只需要这个$$\hat{y_i}$$的预测值比其他的都大即可，而平方损失则过于严格

我们可以采用交叉熵作为损失函数

对于单个的训练样本而言，交叉熵为
$$
-\sum_{i=1}^{输出单元数}第i个实际值*log(第i个预测值)\\
即\\
-\sum_{i=1}^{10}yilog(\hat{y}_i)
$$
此时对于一个训练批次而言，损失函数为
$$
Loss=-\frac{1}{batchsize}\sum_{j=1}^{batchsize}{\sum_{i=1}^{输出单元数}yilog(\hat{y}_i)}
$$
这里就是
$$
Loss=-\frac{1}{64}\sum_{j=1}^{64}{\sum_{i=1}^{10}yilog(\hat{y}_i)}
$$
然而实际上$$y_1,y_2...y_{10}$$这10个实际值只会有一个是1，其他全是0，也就是说，式子变成了
$$
Loss=-\frac{1}{64}\sum_{j=1}^{64}{log(\hat{y}_i)}
$$
然后再结合模型

$$output_i=W_{1,i}*x_1+W_{2,i}*x_2+...+W_{783,i}*x_{783,i}+W_{784,i}*x_{784,i}+b_{i}$$

$$\hat{y_j}=\frac{exp(output_j)}{\sum_{i=1}^{10}{exp(output_i)}}$$

最终损失函数为

$$Loss=-\frac{1}{64}\sum_{j=1}^{64}{log(\frac{exp(output_j)}{\sum_{i=1}^{10}{exp(output_i)}})}$$

## 5.梯度下降优化参数

此时需要用梯度下降优化参数
$$
W_i=W_i-\frac{\eta}{batchsize}\sum_{i=1}^{batchsize}{\frac{d_{Loss}}{dw_i}}\\
b_i=b_i-\frac{\eta}{batchsize}\sum_{i=1}^{batchsize}{\frac{d_{Loss}}{db_i}}
$$


其中$$\eta$$为学习速率，batchsize是批量大小，$$\frac{d_{Loss}}{dw_i}$$为$$w_i$$对损失函数的偏导

#### 假设此时$$\hat{y}$$的对应的类别是预测对的

即按照链式求导法则
$$
\frac{d_{Loss}}{dw_i}\\
=\frac{d_{Loss}}{d_\hat{y}}
\frac{d_\hat{y}}{d_{output_i}}
\frac{d_{output_i}}{d_{W_i}}\\
=\frac{d_{log(\hat{y}_i)}}{d_\hat{y}}
\frac{d_{-\frac{exp(output_i)}{\sum_{j=1}^{10}{exp(output_j)}}}}{d_{output_i}}
\frac{W_{1,i}*x_1+W_{2,i}*x_2+...+W_{783,i}*x_{783,i}+W_{784,i}*x_{784,i}+b_{i}}{d_{W_i}}\\
第二个分式用了分式求导法则,即(\frac{f(x)}{g(x)})^‘=...\\
=\frac{1}{\hat{y}}
\frac{(e^{output_i})*(\sum_{j=1}^{10}{e^{output_j}}-e^{output_i})}{{(\sum_{j=1}^{10}{e^{output_j}}})^2}
x_i\\
=\frac{1}{\frac{e^{output_i}}{\sum_{j=1}^{10}{e^{output_j}}}}
\frac{(e^{output_i})*(\sum_{j=1}^{10}{e^{output_j}}-e^{output_i})}{{(\sum_{j=1}^{10}{e^{output_j}}})^2}
x_i\\
=\frac{(\sum_{j=1}^{10}{e^{output_j}})^2}{e^{output_i}}
\frac{(e^{output_i})*(e^{output_i}-\sum_{j=1}^{10}{e^{output_j}})}{{(\sum_{j=1}^{10}{e^{output_j}}})^2}
x_i\\
消去相同的元素\\
=\frac{e^{output_i}-\sum_{j=1}^{10}{e^{output_j}}}
{\sum_{j=1}^{10}{e^{output_j}}}*x_i\\
由\hat{y_j}=\frac{exp(output_j)}{\sum_{i=1}^{10}{exp(output_i)}}可知\\
原式=(\hat{y}-1)*x_i
$$

因此，更新函数变成了

$$W_i=W_i-\frac{\eta}{batchsize}\sum_{i=1}^{batchsize}{(\hat{y}-1)*x_i}$$

这里就是

$$W_i=W_i-\frac{\eta}{64}\sum_{i=1}^{64}{(\hat{y}-1)*x_i}$$

同理

$$b_i=b_i-\frac{\eta}{64}\sum_{i=1}^{64}{(\hat{y}-1)}$$

#### 同理，假设此时$$\hat{y}$$的对应的类别是预测错误的

$$\frac{d_{Loss}}{dw_i}=\hat{y}*x_i$$

$$W_i=W_i-\frac{\eta}{64}\sum_{i=1}^{64}{\hat{y}*x_i}$$

$$b_i=b_i-\frac{\eta}{64}\sum_{i=1}^{64}{\hat{y}}$$

(推算过程我就省略了，Latex打上去太累了)

PS:我吐了，上面这个求导我一开始没用链式求导，直接手算，$$output_i$$没经过softmax变换，直接当成了$$\hat{y}$$来算，最后结果完全不符合逻辑，半夜看了好多博客终于想起来自己的问题了，好不容易重新推了出来

因此，最终W和b的优化函数如下:

$$W_i=W_i-\frac{\eta}{64}\sum_{i=1}^{64}{(\hat{y}-y)*x_i}$$

$$b_i=b_i-\frac{\eta}{64}\sum_{i=1}^{64}{(\hat{y}-y)}$$

最终代码如下:

```python
w_,b_=CrossEntropyLoss(data,target,y_hat)
w=w+w_  #优化参数w的矩阵
b=b+b_ #优化参数b的矩阵
def CrossEntropyLoss(data,target,y_hat):
    target = np.eye(10)[target]  # 改为one-hot形式
    data=data.T #(784,64)
    #y_hat-target为预测值与实际值的one-hot挨个做差，得出一个(64,10)的y_hat-y的矩阵
    w_=-np.dot(data,y_hat-target)/batch_size*learning_rate
    b_=-(y_hat-target)/batch_size*learning_rate

    # print(w_)
    return w_,b_
```

这短短12行代码我查公式整理原理，搞了一个晚上。。。

最后优化时采用了矩阵相乘的方式代替累加

即$$W_i=W_i-\frac{\eta}{64}\sum_{i=1}^{64}{(\hat{y}-y)*x_i}$$中的$$\sum_{i=1}^{64}{(\hat{y}-y)*x_i}$$

直接用矩阵相乘的方法代替
$$
\left[

 \matrix{  

x_{1,1} & x_{1,2} & ... &x_{1,64} \\  
x_{2,1} & x_{2,2} & ... &x_{2,64} \\ 
x_{3,1} & x_{3,2} & ... &x_{3,64} \\
&  & ...& &\\
x_{782,1} & x_{782,2} & ... &x_{782,64} \\ 
x_{783,1} & x_{783,2} & ... &x_{783,64} \\ 
x_{784,1} & x_{784,2} & ... &x_{784,64} \\  } 

\right]_{shape为(784,64)}*(
\left[

 \matrix{  
\hat{y}_{1,1} &\hat{y}_{2,1} & ... &\hat{y}_{10,1} \\  
\hat{y}_{1,1} & \hat{y}_{2,1} & ... &\hat{y}_{10,1} \\ 
&  & ...& &\\
\hat{y}_{1,1} & \hat{y}_{2,1} & ... &\hat{y}_{10,1} \\ 
\hat{y}_{1,1} & \hat{y}_{2,1} & ... &\hat{y}_{10,1} \\  } 

\right]_{shape为(64，10)}-\left[

 \matrix{  
y_{1,1} & y_{2,1} & ... &x_{10,1} \\  
y_{1,1} & y_{2,1} & ... &x_{10,1} \\ 
&  & ...& &\\
y_{1,1} & y_{2,1} & ... &x_{10,1} \\ 
y_{1,1} & y_{2,1} & ... &x_{10,1} \\  } 

\right]_{shape为(64，10)})
$$
这里相乘时，为该矩阵的每行元素
$$
\left[

 \matrix{  

x_{1,1} & x_{1,2} & ... &x_{1,64} \\  
x_{2,1} & x_{2,2} & ... &x_{2,64} \\ 
x_{3,1} & x_{3,2} & ... &x_{3,64} \\
&  & ...& &\\
x_{782,1} & x_{782,2} & ... &x_{782,64} \\ 
x_{783,1} & x_{783,2} & ... &x_{783,64} \\ 
x_{784,1} & x_{784,2} & ... &x_{784,64} \\  } 

\right]_{shape为(784,64)}
$$
与另一矩阵的每列的元素，挨个相加并且求和

如:$$x_{1,1}*(\hat{y}_{1,1}-y_{1,1})+x_{1,2}(\hat{y}_{1,2}-y_{1,2})+...x_{1,63}(\hat{y}_{1,63}-y_{1,63})+x_{1,64}(\hat{y}_{1,64}-y_{1,64})$$

也就一一对应着$$\sum_{i=1}^{64}{(\hat{y}-y)*x_i}$$

## 计算准确率

```python
def Accuracy(target,y_hat):
    #y_hat.argmax(axis=1)==target 用于比较y_hat与target的每个元素，返回一个布尔数组
    acc=y_hat.argmax(axis=1) == target
    acc=acc+0  #将布尔数组转为0，1数组
    return acc.mean() #通过求均值算出准确率
```



## 最终整个项目完整代码如下

```python
from torchvision import datasets, transforms
import torch.utils.data as Data
import numpy as np
from tqdm._tqdm import trange
batch_size = 64
learning_rate=0.0001

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
            w=GradientDescent(w,w_)  #梯度下降优化参数w
            b=GradientDescent(b,b_)  #梯度下降优化参数b
            list.append(Accuracy(target, y_hat))
            if (batch_idx == 50):
                print(Accuracy(target, y_hat))

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
```

## 结果如下

![image][link2]
















[link1]: data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAvUAAAE1CAYAAABnQh+kAAAgAElEQVR4Xux9C5QdVZX2PrcfiZAOkAyBDAQMRBA6nXRV3U4MCAZUSCKEGWbA8FyOv6AoIqiIqICCqIAiAioCv/pDAxFmUIKSFhEiDDRJblU16QQEguE1AwSTQDpiJ+mu8699rWoqldvd91GPU3W/WuuuJtyqffb5vl11vzpnn30E4QACQAAIAAEgAASAABAAAkAg1QiIVHsP54EAEAACQAAIAAEgAASAABAgiHoEARAAAkAACAABIAAEgAAQSDkCEPUpJxDuAwEgAASAABAAAkAACAABiHrEABAAAkAACAABIAAEgAAQSDkCEPUpJxDuAwEgAASAABAAAkAACAABiHrEABAAAkAACAABIAAEgAAQSDkCEPUpJxDuAwEgAASAABAAAkAACAABiHrEABAAAkAACAABIAAEgAAQSDkCEPUpJxDuAwEgAASAABAAAkAACAABiHrEABAAAkAACAABIAAEgAAQSDkCEPUpJxDuAwEgAASAABAAAkAACAABiHrEABAAAkAACAABIAAEgAAQSDkCEPUpJxDuAwEgAASAABAAAkAACAABiHrEABAAAkAACAABIAAEgAAQSDkCEPUpJxDuAwEgAASAABAAAkAACAABiHrEABAAAkAACAABIAAEgAAQSDkCEPUpJxDuAwEgAASAABAAAkAACAABiHrEABAAAkAACAABIAAEgAAQSDkCEPUpJxDuAwEgAASAABAAAkAACAABiHrEABAAAkAACAABIAAEgAAQSDkCEPUpJxDuAwEgAASAABAAAkAACAABiHrEABAAAkAACAABIAAEgAAQSDkCEPUpJxDuAwEgAASAABAAAkAACAABiHrEQGoQ0DTtSCHEbCKaJKXcM5fL7SmlnEREe7of7sub/BFCrHcch//yv9dLKZfbtv1oajpbB46CzzogGV0cFgHEv9rBAX7U5gfelUYAoh6RoSwC7e3tHblcbq6Ucq4QYi4R7VKjs+9IKZcJIZY5jrOsp6dnZY32cHkFCIDPCsDCqZlDAPGvNqXgR21+4F15CEDUl4cTzooJAU3TZgohPkdE/0pE/+Rvdtq0aZTP52nSpEm0xx577PCZMGFC8dSNGzfSpk2bdvisX7+eCoUCrV27NtiLvxLRr6WUP7Zt+6mYulhXzYDP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
[link2]:data:image/png;base64,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JScna9myZUpMTJQkZWVlOcwJmJSUpGXLlmnmzJl6/vnnFR8fr2effVZXX321t04BAACgQQTMPIDewDxCAAD4H76/A+gWMAAAANxDAAQAADAZAiAAAIDJEAABAABMJmBGAXtD+fgZloQDAMB/lH9vm3kcLAGwDgoKCiSJJeEAAPBDBQUFslqt3q6GVzANTB2Ulpbq8OHDatasWZVrDtdGfn6+EhISdODAAdMOUfcmrr93cf29i+vvfbwHnmUYhgoKChQfH6+gIHP2hqMFsA6CgoLUtm1bj74GS855F9ffu7j+3sX19z7eA88xa8tfOXPGXgAAABMjAAIAAJhM8GOPPfaYtysB14KDgzVq1CiFhHCn3hu4/t7F9fcurr/38R7AkxgEAgAAYDLcAgYAADAZAiAAAIDJEAABAABMhgAIAABgMgRAH7Rw4UIlJSUpIiJCKSkpWr16KTm95gAACFxJREFUtber5PPS09M1YMAANWvWTDExMbryyiv1ww8/OJQxDEOPPfaY4uPj1ahRI40aNUrbt293KFNYWKi7775b0dHRatKkia644godPHjQoczx48d14403ymq1ymq16sYbb9SJEyccymRmZuryyy9XkyZNFB0drenTp6uoqMgzJ++D0tPTZbFYNGPGDNs2rr9nHTp0SDfccIOioqLUuHFj9enTRxs3brQ9z/X3nOLiYj388MNKSkpSo0aN1KFDB82bN0+lpaW2Mlx/+BwDPuXNN980QkNDjb///e/Gjh07jHvuucdo0qSJsX//fm9XzaeNHz/eePnll43vvvvO2Lx5szFx4kSjXbt2xsmTJ21l5s+fbzRr1sx49913jW3bthmTJk0yWrdubeTn59vKTJs2zWjTpo2RkZFhfPvtt8bo0aON3r17G8XFxbYyl1xyiZGcnGysXbvWWLt2rZGcnGxcdtlltueLi4uN5ORkY/To0ca3335rZGRkGPHx8cZdd93VMBfDy77++mujffv2Rq9evYx77rnHtp3r7znHjh0zEhMTjZtuuslYv369sXfvXmPFihXG7t27bWW4/p7z5JNPGlFRUcYHH3xg7N2713j77beNpk2bGgsWLLCV4frD1xAAfczAgQONadOmOWzr2rWr8eCDD3qpRv4pJyfHkGSsXLnSMAzDKC0tNeLi4oz58+fbypw9e9awWq3G4sWLDcMwjBMnThih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