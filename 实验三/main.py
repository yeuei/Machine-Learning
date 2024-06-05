import numpy as np
import matplotlib.pyplot as plt

class logistic_regress:
    def __init__(self, dataset):
        x = dataset[...,:-1] #获得输入
        len = x.shape[0]
        self.y = dataset[...,-1:] #输出向量
        self.x = np.c_[x, np.ones((len,1))] #输入的样本矩阵
        self.feauture_num = self.x.shape[1]
        self.sample_num = self.x.shape[0]
        self.w = np.zeros((self.feauture_num, 1)) #初始化参数向量
        self.lr = 0.1 #默认学习率为0.1
        self.myloss_record = [] #记录损失
        # print(self.y)
        #print(self.x)
        #print(self.feauture_num)
        #print(self.sample_num)
    def func(self,x):
        return 1 / (1 + np.exp(-np.dot(x,self.w)))
    def pre(self,w):#用于预测结果
        return 1 / (1 + np.exp(-np.dot(self.x, w)))
    def loss(self):
        loss = -1 / self.sample_num * (np.sum(self.y * np.log(self.func(self.x)) + (1 - self.y) * np.log(1 - self.func(self.x))))
        return loss
    def loss_grad(self):
        grad_w = -1 / self.sample_num * (np.sum(  (self.y - self.func(self.x)) * self.x, axis=0)).reshape(-1,1) #变为列向量
        return grad_w
    def start_train(self, lr, epoch):#学习率
        self.lr = lr
        for i in range(epoch):
            #第i轮梯度下降
            self.w -= self.loss_grad()*lr
            #计算第i轮损失
            loss_i = self.loss()
            #保存第i轮损失
            self.myloss_record.append(loss_i)
if __name__ == '__main__':
    fpath_test = r'.\experiment_03_testing_set.csv'
    fpath_train = r'.\experiment_03_training_set.csv'

    train_data = np.loadtxt(fpath_train, delimiter=',')
    test_data = np.loadtxt(fpath_test, delimiter=',')
    model = logistic_regress(train_data)
    print(model.loss())
    print(model.loss_grad())
    print('------------\n和老师ppt结果一致开始训练')
    model.start_train(lr = 0.1, epoch = 100)


    #计算预测结果
    test_model = logistic_regress(test_data)
    ans = test_model.pre(model.w)
    pre_ans = np.zeros((test_model.sample_num,1),dtype=np.int_)
    true_ans = test_model.y
    for i in range(test_model.sample_num):
        if ans[i] <= 0.5:
            pre_ans[i] = 0
        else:
            pre_ans[i] = 1
    #获得混淆矩阵
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i in range(test_model.sample_num):
        if(pre_ans[i] == 1):
            if(pre_ans[i] == true_ans[i]):
                TP += 1
            else:
                FP += 1
        else:
            if(pre_ans[i] == true_ans[i]):
                TN += 1
            else:
                FN += 1
    marix = np.array([[TP, FN],
                     [FP, TN]])

    error_rate = (FN + FP)/test_model.sample_num
    accuracy = 1 - error_rate
    precision = TP/(TP + FP)
    recall = TP / (TP + FN)
    F1 = 2 / (1 / precision + 1 / recall)
    print(f'混淆矩阵为:\n{marix}')
    print('错误率为:{:.6f}'.format(error_rate))
    print('精度为:{:.6f}'.format(accuracy))
    print('查准率为:{:.6f}'.format(precision))
    print('查全率为:{:.6f}'.format(recall))
    print('F1为:{:.6f}'.format(F1))
    # print('最优w:')
    # print(model.w)
    #使用中文
    plt.rcParams["font.family"] = "SimHei"
    #将训练时的loss绘制
    fig = plt.figure(num=1, figsize=(10, 8))
    ax = fig.subplots()
    ax.plot(np.arange(1,101),model.myloss_record , color='red', marker='.', linestyle='-', label='学习率:0.1,迭代次数:100')
    plt.legend()
    ax.set_title('实验3：损失曲线迭代图', fontsize=20)
    ax.set_xlabel('训练轮数epoch/轮',size = 15)
    ax.set_ylabel('损失值loss',size = 15)
    plt.savefig('./实验3：损失曲线迭代图.png')
    plt.show()