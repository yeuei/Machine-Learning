import numpy as np
import matplotlib.pyplot as plt
#使用中文
plt.rcParams["font.family"]="SimHei"

fpath_test = r'./experiment_02_testing_set.csv'
fpath_train = r'./experiment_02_training_set.csv'
data_test = np.loadtxt(fpath_test, delimiter = ',')
data_train = np.loadtxt(fpath_train, delimiter = ',')

def func(w, x):
    ans =  np.dot(x,w)
    return np.sum(ans, axis = 1)
def get_w(x, y):
    return np.dot(np.dot(np.linalg.inv(np.dot(x.T, x)), x.T), y)
def make_x(x):#输入为列向量
    my_ones = np.ones_like(x)
    return np.c_[x** 2, x, my_ones]

#获得样本矩阵和输出向量
X = data_train[...,0:1]
Y = data_train[...,1:2]
#定义和train样本数量一样长的向量1
ones = np.ones_like(X)

#生成含x^2, 1的样本矩阵
X = np.c_[X**2,X,ones]
W = get_w(X, Y)
print('参数W为:')
print(get_w(X,Y))

#获得测试矩阵和输出向量
test_X = data_test[...,0:1]
test_Y = data_test[...,1:2]

test_ones = np.ones_like(test_X)
test_X = np.c_[test_X**2,test_X, test_ones]
pre_Y = func(W, test_X)
print('均方误差为:')

MSE = np.sum((pre_Y.reshape(-1,1) - test_Y)**2, axis = 0) / pre_Y.shape[0]
print(MSE)
limit_x = np.array([[min(test_X[...,0])**2,min(test_X[...,0]),1],[max(test_X[...,0])**2,max(test_X[...,1]),1]])

model_limit_x = np.array([min(test_X[...,0]), max(test_X[...,0])])
model_limit_y = func(W, limit_x)


fig = plt.figure(num = 1, figsize=(10,8))
ax = fig.subplots()
ax.plot(data_train[...,0], data_train[...,1], color = 'yellowgreen', marker = 'o', linestyle = '',label = '训练数据')
ax.plot(data_test[...,0], data_test[...,1], color = 'blue', marker = '*',linestyle = '', label = '测试数据')
ax.plot(np.linspace(0,1,1000).reshape(-1,1), func(W, make_x(np.linspace(0,1,1000).reshape(-1,1))) , color = 'red', marker = '.',markersize = 1, linestyle = '-', label = '训练模型')
plt.legend()
ax.set_title('问题2：二次线性模型', fontsize = 20)
ax.set_xlabel('输入X')
ax.set_ylabel('输出Y')
plt.savefig('./第二问画图结果.png')
plt.show()
#print(data_test)
#print(data_train)
