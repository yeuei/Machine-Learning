import numpy as np
import pandas as pd

class Naive_Bayes_Classifiers():
    def load_data(self, path_test = './experiment_07_testing_set.csv', path_train = './experiment_07_training_set.csv'):
        # MyTypes = np.dtype([('Id',np.int_),('SepalLengthCm', np.float_),('SepalWidthCm', np.float_),('PetalLengthCm', np.float_), ('PetalWidthCm',np.float_),('Species','S20')])
        # 记录特征
        self.feauture = ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']
        # 读取数据
        self.train_data = pd.read_csv(path_train)
        self.test_data = pd.read_csv(path_test)
        # 字符串映射
        self.reflection = {'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2}
        self.train_data['Species'] = self.train_data['Species'].replace(['Iris-setosa','Iris-versicolor','Iris-virginica'],[0,1,2])
        self.test_data['Species'] = self.test_data['Species'].replace(['Iris-setosa','Iris-versicolor','Iris-virginica'], [0,1,2])
        # test_data_num
        self.test_data_num = self.test_data.shape[0]
    def cal_priori(self):# 计算先验概率
        self.pri = [0,0,0]
        self.pri[0] = np.mean(self.train_data['Species'] == 0)
        self.pri[1] = np.mean(self.train_data['Species'] == 1)
        self.pri[2] = np.mean(self.train_data['Species'] == 2)
        print('先验概率分别为:')
        for i, Y in enumerate(self.reflection.keys()):
            print(f'{Y}的先验概率: {self.pri[i]}')
    def cal_condition(self):# 计算条件概率
        self.condition_ans = [[] for _ in range(3)]
        for i, Y in enumerate(self.reflection.keys()):
            idx = self.train_data['Species'] == self.reflection[Y]
            tmp_total = self.train_data[idx]
            for X in self.feauture:
                tmp = tmp_total[X].to_numpy()
                ave = np.mean(tmp)
                var = np.sqrt(np.var(tmp))
                self.condition_ans[i].append([round(ave, 6), round(var, 6)])
                # print(f'P({X} = x|Y = {Y})')
        print('高斯分布参数估计为:')
        for item in self.condition_ans:
            print(item)
    def get_p(self, x, condition_list):
        ave = condition_list[0]
        var = condition_list[1]
        return 1 / (np.sqrt(2 * np.pi) * var) * np.exp(-(x - ave)**2 / (2 * var**2))
    def get_acc(self):
        right = 0
        # print(self.test_data.shape[0])
        for i in range(1, self.test_data_num + 1):
            id = self.test_data['Id'] == i
            tmp = self.test_data[id]
            #初始化预测
            best_pre = 0 #预测值为0
            best_pre_ans = 0 #第0个
            for i, Y in enumerate(self.reflection.keys()):
                p = self.pri[i]
                for j,X in enumerate(self.feauture):
                    p *= self.get_p(x = tmp[X].to_numpy(), condition_list = self.condition_ans[i][j])
                if p > best_pre:
                    best_pre_ans = i
                    best_pre = p
            if(best_pre_ans == tmp['Species'].to_numpy()):
                right += 1
        print(f'精度为:{right/self.test_data_num}')
        return right/self.test_data_num
if __name__ == '__main__':
    # 生成模型
    NBC_Model = Naive_Bayes_Classifiers()

    # 载入数据
    NBC_Model.load_data()

    # 计算先验概率
    NBC_Model.cal_priori()

    # 计算条件概率
    NBC_Model.cal_condition()

    # 计算测试集的准确率
    NBC_Model.get_acc()

    # 测试和为1
    # print(f'{NBC_Model.pri[0]} + {NBC_Model.pri[1]} + {NBC_Model.pri[2]} == {NBC_Model.pri[0] + NBC_Model.pri[1] + NBC_Model.pri[2]}')
