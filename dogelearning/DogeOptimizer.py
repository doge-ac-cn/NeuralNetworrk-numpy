import numpy as np
class Optimizer :

    learning_rate=0.1

    def __init__(self,learning_rate=0.1):
        self . learning_rate=learning_rate

    def update(self, param_name, param, gradient_param, gamma=0.9):
        pass

class SGDOptimizer(Optimizer):
    def update(self,param_name,param,gradient_param):
        param=param - gradient_param * self.learning_rate
        return param

class MomentumOptimizer(Optimizer):

    def __init__(self, param_names,param_shapes,learning_rate=0.1):
        #传入参数名和参数数量，从而初始化参数各自的学习速率矩阵
        self.gradient_cache= {}
        self.learning_rate = learning_rate
        for i in  range (0,len(param_names)):
            self.gradient_cache[param_names[i]]=np.zeros(param_shapes[i])

    def update(self,param_name,param,gradient_param,gamma=0.9):
        #gamma用于调整学习率


        self.gradient_cache[param_name] = self.gradient_cache[param_name] * gamma + gradient_param * (1 - gamma)
        #优化梯度矩阵并且缓存
        param= param - self.gradient_cache[param_name]*self.learning_rate
        #优化参数的学习速率矩阵
        return param

class AdagradOptimizer(Optimizer):
    def update(self,param,gradient_param):
        param= param - self.learning_rate*gradient_param
        return param

