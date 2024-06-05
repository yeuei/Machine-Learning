import numpy as np
from sklearn import tree
import matplotlib.pyplot as plt

class RandomForest:
    def __init__(self, n): #基决策树的个数n
        self.n = n
        self.BaseTree = [tree.DecisionTreeClassifier(random_state=1,criterion='entropy',max_features = 50) for _ in range(n)]
        np.random.seed(100)
    def load_data(self,path_train = './experiment_09_training_set.csv', path_test = './experiment_09_testing_set.csv'):
        # 载入数据集
        self.train_data = np.loadtxt(fname = path_train, delimiter = ',', skiprows = 1)
        self.test_data = np.loadtxt(fname = path_test, delimiter = ',', skiprows = 1)
        # 自主采样抽取次数m
        self.m = self.train_data.shape[0]

        # 分割输入输出
        self.train_x = self.train_data[:,1:]
        self.train_y = self.train_data[:, 0]

        self.test_x = self.test_data[:,1:]
        self.test_y = self.test_data[:, 0]

        # 获取测试集个数
        self.test_num = self.test_data.shape[0]
    def train(self): #训练
        for T in range(self.n):
            sample_ans = np.random.choice(self.m, self.m, replace=True)
            X = self.train_x[sample_ans]
            Y = self.train_y[sample_ans]
            self.BaseTree[T].fit(X, Y)
    def get_acc(self): #获得精度结果
        ans_matrix = np.zeros((self.test_num, 10)) #第i行表示第i个样本,10列表示0,1,2,3~9被预测的次数
        for j in range(self.n): # 第j个决策树
            ind1 = self.BaseTree[j].predict(self.test_x).reshape(1, -1) #获得第j个决策树对所有测试集的预测结果向量
            # print(tmp.shape) #(12000,) 第i个值表示：第i个测试集被第j个决策树预测的值

            # 为索引，故而转换为整数类型
            ind1 = ind1.astype(dtype = np.int_)
            ind2 = np.arange(self.test_num, dtype = np.int_)
            ans_matrix[ind2, ind1] += 1
        # 找到每行最大值的索引，也就是预测标签
        max_indices = np.argmax(ans_matrix, axis=1)
        # print(max_indices)
        # print(max_indices.shape)
        acc = np.mean(max_indices == self.test_y)
        return acc
def draw(n:int, accs:list):
    #使用中文
    plt.rcParams["font.family"] = "SimHei"
    #将训练时的loss绘制
    fig = plt.figure(num=1, figsize=(10, 8))
    ax = fig.subplots()
    ax.plot(np.arange(1, n + 1), accs, color='red', marker='.', linestyle='-', label='随机选取属性子集数目为50的精度变化曲线')
    plt.legend()
    ax.set_title('实验9：精度随决策树数量T增加的变化曲线', fontsize=20)
    ax.set_xlabel('决策树数量T(number of decision trees)',size = 15)
    ax.set_ylabel('精度(accuracy)',size = 15)
    #设置横坐标精度为1
    xticks = np.arange(1, n + 1)
    ax.set_xticks(xticks[::1])
    #保存
    plt.savefig('./实验9：精度随决策树数量T增加的变化曲线.png')
    plt.show()
if __name__ == '__main__':
    n = 20 # T的最大值
    accs = [] # 保存精度
    for T in range(1, n+1): #遍历1~20
        # 初始化模型
        MyModel = RandomForest(T)
        # 载入数据
        MyModel.load_data()
        # 开始训练
        MyModel.train()
        # 计算精确度
        acc = MyModel.get_acc()
        print(f'T = {T}时，acc = {acc:.4f}')
        accs.append(acc)
    draw(n, accs)
