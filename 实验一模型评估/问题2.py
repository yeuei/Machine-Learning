import numpy as np
fpath1 = './experiment_01_dataset_01.csv'
fpath2 = './experiment_01_dataset_02.csv'
#问题2
data2 = np.loadtxt(fpath2, delimiter = ',')
#print(data2)
y2_true = data2[:, 1]
y2_pre = []
for i in range(3):
    y2_pre.append(data2[:, i + 2])
TP = np.zeros(3)
FN = np.zeros(3)
FP = np.zeros(3)
TN = np.zeros(3)
marix = [np.zeros((2,2),dtype=int) for _ in range(3)]#保存三个混淆矩阵
total_len = np.size(y2_true, axis = 0)
for i in range(total_len):
    for j in range(3):
        if(y2_pre[j][i] == 0):
            if(y2_true[i] == 0):
                TN[j] += 1
            else:
                FN[j] += 1
        if(y2_pre[j][i] == 1):
            if(y2_true[i] == 0):
                FP[j] += 1
            else:
                TP[j] += 1
for i in range(3):
    marix[i][0][0] = TP[i]
    marix[i][0][1] = FN[i]
    marix[i][1][0] = FP[i]
    marix[i][1][1] = TN[i]
    print(f'第{i+1}个混淆矩阵为:')
    print(marix[i])

precision = []
recall = []
F1 = []
for i in range(3):
    precision.append((marix[i][0][0]) / (marix[i][0][0] + marix[i][1][0]))
    recall.append((marix[i][0][0]) / (marix[i][0][0] + marix[i][0][1]))
    F1.append(2 / (1/precision[i] + 1/recall[i]))
    print('第%d个的precision是%f%%,recall是%f%%,F1分数是%f%%' % (i+1, precision[i]*100, recall[i]*100, F1[i]*100))
