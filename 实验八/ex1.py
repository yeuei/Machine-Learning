import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree

class AdaBoosting:
    def __init__(self, n = 1):#n 为基训练器的个数，默认为1
        # 保存 n
        self.n = n
        # 生成n个基训练器
        self.BaseModel = [tree.DecisionTreeClassifier(random_state=1, criterion='entropy', max_depth=3) for _ in range(n)]
        self.a = [0 for _ in range(n)]
    def load_data(self, test_path = './experiment_08_testing_set.csv', train_path = './experiment_08_training_set.csv'):
        # 读入数据
        self.test_data = np.loadtxt(fname = test_path, delimiter = ',')
        self.train_data = np.loadtxt(fname = train_path, delimiter = ',')

        # 分测试数据的输入和输出
        self.test_data_x = self.test_data[:,:-1]
        self.test_data_y = self.test_data[:,-1]

        # 分训练数据的输入和输出
        self.train_data_x = self.train_data[:,:-1]
        self.train_data_y = self.train_data[:,-1]
        # 获得样本数量
        self.sample_num = self.train_data.shape[0]

        # 获得测试集的数量
        self.test_num = self.test_data.shape[0]

        # 输出样本数量
        # print(self.sample_num)
    def train(self):
        # 初始化样本权重 W
        self.W = np.full(self.sample_num, 1 / self.sample_num)

        for i in range(self.n):
            # 训练
            self.BaseModel[i].fit(self.train_data_x,self.train_data_y,self.W)
            # 计算错误率
            e = np.sum((self.BaseModel[i].predict(self.train_data_x) != self.train_data_y) * self.W)
            # 计算权重
            self.a[i] = 0.5 * np.log((1-e) / e)
            # 更新样本权重
            tmp = self.BaseModel[i].predict(self.train_data_x) * self.train_data_y
            self.W *= np.exp(-tmp * self.a[i])
            # 归一化样本权重
            self.W = self.W / np.sum(self.W)

    def get_acc(self):
        ans = np.zeros(self.test_num)
        for i in range(self.n):
            ans += self.BaseModel[i].predict(self.test_data_x) * self.a[i]
        Final_Ans = np.sign(ans)
        return np.mean(Final_Ans == self.test_data_y)
def draw(n:int, accs:list):
    #使用中文
    plt.rcParams["font.family"] = "SimHei"
    #将训练时的loss绘制
    fig = plt.figure(num=1, figsize=(10, 8))
    ax = fig.subplots()
    ax.plot(range(1,n+1), accs, color='red', marker='.', linestyle='-', label='精度变化曲线')
    plt.legend()
    ax.set_title('实验8：精度随T增加的变化曲线', fontsize=20)
    ax.set_xlabel('基模型个数T(number of base models)',size = 15)
    ax.set_ylabel('精度(accuracy)',size = 15)
    ax.plot(np.arange(1,n+1), accs, color='red', marker='.', linestyle='-', label='精度变化曲线')
    #设置横坐标精度为1
    xticks = np.arange(1, n + 1)
    ax.set_xticks(xticks[::1])
    #保存
    plt.savefig('./实验8：精度随T增加的变化曲线.png')
    plt.show()
if __name__ == '__main__':
    n = 20
    #保存每种模型的精度
    accs = []
    for i in range(1, n + 1):
        # 生成n个基模型
        MyModel = AdaBoosting(n = i)
        # 载入数据
        MyModel.load_data()
        # 开始训练
        MyModel.train()
        # 获取精度
        acc = MyModel.get_acc()
        # 输出精度
        print(f'第{i}个集成模型的精确度为{acc:.4f}')
        # 保存精度
        accs.append(acc)
    draw(n = n, accs = accs)
