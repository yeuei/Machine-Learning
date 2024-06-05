from scipy.optimize import LinearConstraint, minimize
import numpy as np
import matplotlib.pyplot as plt

class SVM():
    def __init__(self):
        #用类写，便不需要用global定义如下两个变量
        self.iteration = 1
        self.loss_iteration = []
        #only one dimension
        self.w = np.random.randn(3,)
        # print(self.w)
        # print(self.b)
    def load_data(self, path_train = './experiment_06_training_set.csv', path_test = './experiment_06_testing_set.csv'):
        self.test_data = np.loadtxt(path_test, delimiter=',')
        self.train_data = np.loadtxt(path_train, delimiter=',')
        # 测试数据
        self.test_x = self.test_data[:,0:2]
        self.test_y = self.test_data[:,-1].reshape(-1,1)
        # 训练数据
        self.train_x = self.train_data[:, 0:2]
        self.train_y = self.train_data[:, -1].reshape(-1,1)
        self.sample_num = self.train_data.shape[0]
        self.test_num = self.test_y.shape[0]
        # 在末尾添上1
        self.train_true_x = np.c_[self.train_x, np.ones((self.sample_num,1))]
        self.test_true_x = np.c_[self.test_x, np.ones((self.test_num, 1))]
        # print(self.sample_num)
        # print(self.test_x.shape)
        # print(self.test_y.shape)
    # 定义目标函数
    def objective(self, parameter):
        return (parameter[0]**2 + parameter[1]**2) / 2
    # 定义线性约束
    def restraint(self):
        ultimate_A = self.train_y * self.train_true_x #广播yi * xi
        ultimate_lb = np.ones(self.sample_num)
        ultimate_ub = np.ones(self.sample_num)
        ultimate_ub[:] = np.inf
        print(ultimate_lb.shape)
        print(ultimate_A.shape)
        print(ultimate_ub.shape)
        # print(A)
        self.linear_constraint = LinearConstraint(A = ultimate_A, lb = ultimate_lb, ub = ultimate_ub, keep_feasible = False)
    def my_minmize(self):
        self.restraint()
        self.res = minimize(fun = self.objective, x0 = self.w, method='trust-constr', constraints=self.linear_constraint, callback = self.print_loss)
        self.ultimate_w = self.res.x
    def print_loss(self, intermediate_result):
        parameter = intermediate_result.x
        self.loss_iteration.append(self.objective(parameter))
        print('iteration', self.iteration, 'loss', self.objective(parameter))
        self.iteration += 1
    def draw1(self):
        # 使用中文
        plt.rcParams["font.family"] = "SimHei"
        # 正常显示负号
        plt.rcParams["axes.unicode_minus"] = False
        fig = plt.figure(num=1, figsize=(10, 8))
        ax = fig.subplots()
        ax.plot(range(1, len(self.loss_iteration) + 1), self.loss_iteration, color = 'red', marker = '.', linestyle = '-', markersize = 1,label = '损失函数迭代曲线')
        plt.legend()
        ax.set_title(f'实验6：支持向量机-损失函数迭代曲线', fontsize=20)
        ax.set_xlabel('迭代次数Epoch', fontsize=15)
        ax.set_ylabel('损失Loss', fontsize=15)
        plt.savefig(f'./损失迭代曲线.png')
        plt.show()
    def draw2(self):
        # 使用中文
        plt.rcParams["font.family"] = "SimHei"
        data_list = []
        fig = plt.figure(num=1, figsize=(10, 8))
        ax = fig.subplots()
        #ax.plot(range(1, len(self.loss_iteration) + 1), self.loss_iteration, color = 'red', marker = '.', linestyle = '-', markersize = 1,label = '迭代损失曲线')
        # input((self.train_y == -1)[:,0])
        ax.plot(self.train_x[(self.train_y == -1)[:, 0], 0],self.train_x[(self.train_y == -1)[:,0], 1], marker = '.', linestyle = '', markersize = 10, label = '训练集负例',color = 'blue')
        ax.plot(self.train_x[(self.train_y == 1)[:, 0], 0], self.train_x[(self.train_y == 1)[:, 0], 1], marker='.', linestyle='', markersize=10, label='训练集正例', color = 'red')
        min_x = np.min(self.train_x[:, 0])
        max_x = np.max(self.train_x[:, 0])
        X = np.linspace(start = min_x, stop = max_x, num = int(max_x - min_x) * 1000)
        Y = -(self.ultimate_w[0] * X + self.ultimate_w[2])/self.ultimate_w[1]
        ax.plot(X, Y, marker='.', linestyle='-', markersize=3, label='分类超平面', color = 'grey')
        plt.legend()
        ax.set_title(f'实验6：支持向量机-训练集分类超平面图', fontsize=20)
        ax.set_xlabel('X1', fontsize=15)
        ax.set_ylabel('X2', fontsize=15)
        plt.savefig(f'./训练集分类超平面图.png')
        plt.show()
    def draw3(self):
        # 使用中文
        plt.rcParams["font.family"] = "SimHei"
        data_list = []
        fig = plt.figure(num=1, figsize=(10, 8))
        ax = fig.subplots()
        #ax.plot(range(1, len(self.loss_iteration) + 1), self.loss_iteration, color = 'red', marker = '.', linestyle = '-', markersize = 1,label = '迭代损失曲线')
        # input((self.train_y == -1)[:,0])
        ax.plot(self.test_x[(self.test_y == -1)[:, 0], 0],self.test_x[(self.test_y == -1)[:,0], 1], marker = '.', linestyle = '', markersize = 10, label = '测试集负例',color = 'blue')
        ax.plot(self.test_x[(self.test_y == 1)[:, 0], 0], self.test_x[(self.test_y == 1)[:, 0], 1], marker='.', linestyle='', markersize=10, label='测试集正例', color = 'red')
        min_x = np.min(self.test_x[:, 0])
        max_x = np.max(self.test_x[:, 0])
        X = np.linspace(start = min_x, stop = max_x, num = int(max_x - min_x) * 1000)
        Y = -(self.ultimate_w[0] * X + self.ultimate_w[2])/self.ultimate_w[1]
        ax.plot(X, Y, marker='.', linestyle='-', markersize=3, label='分类超平面', color = 'grey')
        plt.legend()
        ax.set_title(f'实验6：支持向量机-测试集分类超平面图', fontsize=20)
        ax.set_xlabel('X1', fontsize=15)
        ax.set_ylabel('X2', fontsize=15)
        plt.savefig(f'./测试集分类超平面图.png')
        plt.show()
    def get_acc(self):
        total_n = self.test_num
        right = 0
        for i in range(total_n):
            ans = np.sum(self.ultimate_w * self.test_true_x[i,:], axis = 0) * self.test_y[i]
            # input(ans)
            if(ans >= 0):
                right += 1
        return right / total_n
if __name__ == '__main__':
    #初始化和载入数据
    SvmModel = SVM()
    SvmModel.load_data()

    #进行训练
    SvmModel.my_minmize()

    # 画迭代损失曲线
    SvmModel.draw1()

    # 输出参数
    print(SvmModel.ultimate_w)

    # 计算精确度
    acc = SvmModel.get_acc()
    print(acc)

    # 画训练集分割图
    SvmModel.draw2()

    #画测试集分割图
    SvmModel.draw3()
