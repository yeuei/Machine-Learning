import numpy as np
fpath1 = './experiment_01_dataset_01.csv'
fpath2 = './experiment_01_dataset_02.csv'
#问题1
data1 = np.loadtxt(fpath1, delimiter = ',')
print(data1)
y1_true = data1[:, 1]
y1_pre = []
for i in range(3):
    y1_pre.append(data1[:, i + 2])

#计算MSE
MSE = [] #计算MSE，prediction1~3分别依次存放在MSE中
MAE = [] #计算MAE，prediction1~3分别依次存放在MAE中
RMSE = [] #计算RMSE，prediction1~3分别依次存放在RMSE中
total_len = np.size(y1_true, axis=0) # 计算行数
for i in range(3):
    uMSE = np.sum((y1_true - y1_pre[i])**2, axis = 0)
    uMAE = np.sum(abs(y1_true - y1_pre[i]), axis = 0)
    uMAE /= total_len
    uMSE /= total_len
    uRMSE = uMSE**(1/2)
    MSE.append(uMSE)
    MAE.append(uMAE)
    RMSE.append(uRMSE)
print(f'MSE是{MSE}')
print(f'MAE是{MAE}')
print(f'RMSE是{RMSE}')

#问题2
data2 = np.loadtxt(fpath2, delimiter = ',')
print(data2)
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
    print(f'第{i+1}个的precision是{precision[i]*100}%,recall是{recall[i]*100}%,F1分数是{F1[i]*100}%')

