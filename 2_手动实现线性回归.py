from IPython import display
from matplotlib import pyplot as plt
from mxnet import autograd, nd
import random


# 读取数据集
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        j = nd.array(indices[i:min(i + batch_size, num_examples)])  # 此处是切片，不整除时会报错
        yield features.take(j), labels.take(j)


# 定义线性回归矢量表达式
def linreg(X, w, b):
    return nd.dot(X, w) + b  # 注意广播机制


# 损失函数
def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


# 定义优化算法
def sgd(params, lr, batch_size):
    for param in params:
        param[:] = param - lr * param.grad / batch_size


# 生成数据集
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = nd.random.normal(scale=1, shape=(num_examples, num_inputs))
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += nd.random.normal(scale=0.01, shape=labels.shape)  # 加入随机噪声

# 初始化参数
w = nd.random.normal(scale=0.01, shape=(num_inputs, 1))
b = nd.zeros(shape=(1,))

# 小批量大小
batch_size = 10

# 创建梯度
w.attach_grad()
b.attach_grad()

lr = 0.03  # 学习率
num_epochs = 3  # 迭代周期数
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        with autograd.record():
            l = loss(net(X, w, b), y)
        l.backward()
        sgd([w, b], lr, batch_size)
    train_loss = loss(net(features, w, b), labels)
    print('epoch %d,loss %f' % (epoch + 1, train_loss.mean().asnumpy()))

print(true_w,w)
print(true_b,b)