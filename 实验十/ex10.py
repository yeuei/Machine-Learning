import numpy as np
import matplotlib.pyplot as plt

class Kmeans():
    def __init__(self, k):
        # 保存超参数：聚类个数k
        self.k = k
        # 保存每次迭代之后的聚类结果
        self.indice = [[] for _ in range(self.k)]
        # 使用固定的随机数种子
        np.random.seed(92)
    def load_data(self, fpath_train = './experiment_10_training_set.csv'):
        # 保存训练样本
        self.train_data = np.loadtxt(fname = fpath_train, delimiter = ',')
        # 保存样本个数
        self.sample_num = self.train_data.shape[0]
        # 生成随机聚类中心
        choice_k = np.random.choice(self.sample_num, self.k, replace = False)
        self.centers = self.train_data[choice_k, :]
    def train(self, epochs = 100):
        # 默认迭代100次，符合报告要求
        for epoch in range(epochs):
            # 初始化每次迭代之后的聚类结果
            self.indice = [[] for _ in range(self.k)]
            for row in range(self.sample_num):
                # 计算距离
                tmp_ans = self.cal_dis_2(self.train_data[row, :])
                # 得到改点到距离最短的中心的编号
                IndMin = np.argmin(tmp_ans)
                # 更新聚类结果
                self.indice[IndMin].append(row)
            # 更新聚类中心
            self.cal_new_center()
        # 计算损失
        self.loss = self.cal_loss()
    def cal_dis_2(self, data):
        return np.sum( (np.tile(data, (self.k,1)) - self.centers)**2, axis = 1)
    def cal_new_center(self): #结束一次迭代之后，更新聚类中心
        for i in range(self.k):
            tmp_group = self.train_data[self.indice[i], :]
            self.centers[i,:] = np.mean(tmp_group, axis = 0)
    def cal_loss(self): #计算损失函数
        loss = 0
        for i in range(self.k):
            tmp = self.train_data[self.indice[i], :]
            loss += np.sum((tmp - self.centers[i])**2)
        # 返回最终损失
        return loss
    def draw_scatter(self):
        # 使用中文
        plt.rcParams["font.family"] = "SimHei"
        # 正常显示负号
        plt.rcParams["axes.unicode_minus"] = False
        # 创建颜色映射表
        cmap = plt.get_cmap('viridis', self.k)
        # 新建figure
        plt.figure(num=1, figsize=(10, 8))
        for i in range(0, self.k):
            plt.scatter(self.train_data[self.indice[i], 0], self.train_data[self.indice[i], 1], color = cmap(i), label = f'第{i + 1}类')
        plt.legend()
        plt.title(f'实验10:k为{self.k}时迭代100次的聚类结果', fontsize=20)
        plt.savefig(f'./实验10：k为{self.k}时迭代100次的聚类结果.png')
        plt.show()
def draw_loss(loss_list, n):
    #使用中文
    plt.rcParams["font.family"] = "SimHei"
    #将训练时的loss绘制
    fig = plt.figure(num=1, figsize=(10, 8))
    ax = fig.subplots()
    ax.plot(np.arange(1, n + 1), loss_list, color='red', marker='.', linestyle='-', label='loss随k值增加的变化曲线')
    plt.legend()
    ax.set_title('实验10:loss随k值增加的变化曲线图', fontsize=20)
    ax.set_xlabel('k值(聚类中心个数)',size = 15)
    ax.set_ylabel('loss(损失值)',size = 15)
    #设置横坐标精度为1
    xticks = np.arange(1, n + 1)
    ax.set_xticks(xticks[::1])
    #保存
    plt.savefig('./实验10：loss随k值增加的变化曲线图.png')
    plt.show()
if __name__ == '__main__':
    n = 10
    loss_list = []
    for k in range(1, n + 1):
        # 初始化模型
        MyModel = Kmeans(k = k)
        # 读取训练集
        MyModel.load_data()
        # 开始迭代次数为100的训练
        MyModel.train(epochs = 100)
        # 保存loss_list结果
        loss_list.append(MyModel.loss)
        # 画聚类结果图
        MyModel.draw_scatter()
    # 画loss随k值增加的变化曲线
    draw_loss(loss_list = loss_list, n = n)
    # 输出loss_list
    print(loss_list)
