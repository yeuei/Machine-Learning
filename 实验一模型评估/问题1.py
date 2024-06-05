import numpy as np
fpath1 = './experiment_01_dataset_01.csv'
fpath2 = './experiment_01_dataset_02.csv'
#问题1
data1 = np.loadtxt(fpath1, delimiter = ',')
#print(data1)
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
#print(f'MSE是{MSE}')
print('MSE是')
for iter in MSE:
    print('%f'%(iter), end = ',')
print()

print('MAE是')
for iter in MAE:
    print('%f' % (iter), end = ',')
print()

print('RMSE是')
for iter in RMSE:
    print('%f' % (iter), end = ',')
