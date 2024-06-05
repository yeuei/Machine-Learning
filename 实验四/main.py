import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree

def read_data(fpath:str):
    datas = np.loadtxt(fname=fpath, delimiter=',')
    return datas[:,:-1], datas[:,-1:]
def cal_accuracy(pre_y, y_test):
    ans = 0
    lens = pre_y.shape[0]
    for i in range(lens):
        if(pre_y[i] == y_test[i]):
            ans += 1
    return ans / lens

if __name__ == "__main__" :
    fpath_test = r'./experiment_04_testing_set.csv'
    fpath_train = r'./experiment_04_training_set.csv'
    x_train, y_train = read_data(fpath_train)
    x_test, y_test = read_data(fpath_test)
    #将输出结果(类型)变为整型
    y_train = np.array(y_train, dtype=np.int_)
    y_test = np.array(y_test, dtype=np.int_)

    #构造class_name

    criterions = ['gini','entropy']
    max_depths = [1, 2, 3]

    cnt = 1
    print("准则/层数",end='\t')
    print("1\t\t\t2\t\t\t3")
    for criterion in criterions:
        print(f"{criterion}",end = '\t')
        for max_depth in max_depths:
            tdc = tree.DecisionTreeClassifier(criterion=criterion, random_state=1, max_depth=max_depth)
            tdc = tdc.fit(x_train, y_train)#进行训练
            pre_y = tdc.predict(x_test)
            accuracy = cal_accuracy(pre_y = pre_y, y_test=y_test)
            print('{:.8f}'.format(accuracy),end='\t')
            #绘制并保存图片
            plt.rcParams["font.family"] = "SimHei"# 使用中文
            fig = plt.figure(num=cnt, figsize=(10, 8))
            cnt += 1
            ax = fig.subplots()
            tree.plot_tree(decision_tree=tdc, filled=True)  # 使用库函数生成决策树图片
            ax.set_title(f'实验4：准则:{criterion}，最大层数:{max_depth}', fontsize=20)
            plt.savefig(f'准则_{criterion}_最大层数_{max_depth}.png')  # 保存决策树图片
        print()
    plt.show()
    #print(x_train)
    #print(y_train)
    #print(x_train.shape[1])

