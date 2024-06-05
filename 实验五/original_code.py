import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

class Fully_Neural():
    def __init__(self):
        self.theta = np.random.randn(10, 1)
        self.V = np.random.randn(10, 12)
        self.b = np.random.randn(12, 1)
        self.W = np.random.randn(12, 784)
    def initial(self):
        self.theta = np.random.randn(10, 1)
        self.V = np.random.randn(10, 12)
        self.b = np.random.randn(12, 1)
        self.W = np.random.randn(12, 784)
    def load_data(self, path_train = 'experiment_05_training_set.csv', path_test = 'experiment_05_testing_set.csv'):
        train_data = pd.read_csv(path_train)
        test_data = pd.read_csv(path_test)
        train_data = np.array(train_data)
        test_data = np.array(test_data)
        #归一化
        self.train_x = train_data[:, 1:] / 255
        self.train_y = train_data[:, 0]
        self.test_x = test_data[:, 1:] / 255 #(m, 784)
        self.test_y = test_data[:, 0] #(m, 1)
        #one-hot
        self.n = self.train_x.shape[0]
        self.m = self.test_x.shape[0]
        self.one_hot_h = np.zeros((self.n, 10))
        #第i个one-hot的label_y行为1，即one-hot
        #问题：
        self.one_hot_h[np.arange(self.n),self.train_y.reshape((self.n,))]  = 1
    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))
    def forward(self, i):
        #x
        self.x = self.train_x[i, :].reshape(784,1) #(784, 1) 问题(784,)不能转置
        #print(self.x)
        #h
        self.h = self.one_hot_h[i,:].reshape(10,1) #(10 * 1)
        #print(self.h.shape)
        #z
        self.z = np.dot(self.W, self.x)
        self.z += self.b #(12, 1)

        #a
        self.a = self.sigmoid(self.z) #(12, 1)

        #t
        self.t = np.dot(self.V, self.a)
        self.t += self.theta # 10 * 1

        #y
        self.y = self.sigmoid(self.t)

        #L 问题：标量
        self.L = np.sum(((self.y - self.h)**2) / 2, axis = 0)

        #返回loss
        return self.L
    def backward(self):
        self.df_theta = (self.y - self.h) * self.y * (1 - self.y)

        self.df_V = np.dot(self.df_theta, self.a.T)

        self.df_b = np.dot((self.V).T, self.df_theta)*(self.a)*(1 - self.a)

        self.df_W = np.dot(self.df_b, (self.x).T)
    def updata(self, lr):
        self.theta -= lr * self.df_theta
        self.V -= lr * self.df_V
        self.b -= lr * self.df_b
        self.W -= lr * self.df_W
    def draw(self, lr, epochs):
        # 使用中文
        plt.rcParams["font.family"] = "SimHei"
        data_list = []
        with open(f'./学习率为{lr}.txt', 'r', encoding='utf-8') as f:
            tmp = f.readline()
            while(tmp != ''):
                tmp = tmp.strip('[').strip('\n').strip(']')
                data_list.append(float(tmp))
                tmp = f.readline()
        # input(data_list)
        fig = plt.figure(num=1, figsize=(10, 8))
        ax = fig.subplots()
        ax.plot(range(1, epochs + 1), data_list, color = 'red', marker = '.', linestyle = '-', markersize = 1,label = f'学习率为{lr}')
        plt.legend()
        ax.set_title(f'实验5：神经网络-学习率为{lr}时损失迭代曲线', fontsize=20)
        ax.set_xlabel('迭代次数Epoch', fontsize=15)
        ax.set_ylabel('损失Loss', fontsize=15)
        plt.savefig(f'./学习率为{lr}时损失迭代曲线.png')
        plt.show()
    def train(self, lr, epochs = 100):
        self.loss_list = []
        for epoch in  range(1, epochs + 1): #100最大轮次
            #获取随机数序列
            random_list = list(range(0, self.n))
            random.shuffle(random_list)
            #开始训练
            total_loss = 0
            for seq in random_list:
                l = self.forward(seq)
                self.backward()
                self.updata(lr)
                total_loss += l
            total_loss /= self.n
            self.loss_list.append(total_loss)
            print(f"第{epoch}轮的loss = {total_loss}")
        #保存数据
        with open(f'./学习率为{lr}.txt', 'w', encoding='utf-8') as f:
            for item in self.loss_list:
                f.write(str(item))
                f.write('\n')
        #画图
        self.draw(lr, epochs)
    def pre(self,x):
        #z
        z = np.dot(self.W, x)
        z += self.b #(12, 1)

        #a
        a = self.sigmoid(z) #(12, 1)

        #t
        t = np.dot(self.V, a)
        t += self.theta # 10 * 1

        #y
        y = self.sigmoid(t)
        return y
    def get_acc(self):
        right = 0
        for i in range(self.m): #m个样本
            ans_hot = self.pre(self.train_x[i, :].reshape(784,1))
            max_value = max(ans_hot)
            #找最大值的下标
            index = np.argmax(ans_hot)
            if(index == self.train_y[i]):
                right += 1
        print(f"精度为:{right/self.m}")
        return right / self.m
if '__main__' == __name__:
    model = Fully_Neural()
    model.load_data()
    # L = model.forward(1)
    # print(L)

    model.initial()
    model.train(lr = 0.001, epochs = 100)
    model.get_acc()

    model.initial()
    model.train(lr = 0.005, epochs = 100)
    model.get_acc()

    model.initial()
    model.train(lr = 0.01, epochs = 100)
    model.get_acc()
