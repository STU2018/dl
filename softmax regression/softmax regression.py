import d2lzh as d2l
from mxnet import autograd, nd

batch_size = 256
num_epochs, lr = 1, 0.1
num_inputs = 784
num_outputs = 10

W = nd.random.normal(scale=0.01, shape=(num_inputs, num_outputs))
b = nd.zeros(num_outputs)
W.attach_grad()
b.attach_grad()


# softmax函数
def softmax(X):
    X_exp = X.exp()
    partition = X_exp.sum(axis=1, keepdims=True)
    return X_exp / partition  # 这⾥应⽤了⼴播机制


def net(X):
    return softmax(nd.dot(X.reshape((-1, num_inputs)), W) + b)


# 交叉熵损失
def cross_entropy(y_hat, y):
    return -nd.pick(y_hat, y).log()  # pick是根据标签y选择y_hat中的概率值


# 评价正确率
def accuracy(y_hat, y):
    return (y_hat.argmax(axis=1) == y.astype('float32')).mean().asscalar()


# 评价模型正确率
def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        y = y.astype('float32')
        acc_sum += (net(X).argmax(axis=1) == y).sum().asscalar()
        n += y.size
    return acc_sum / n


# 训练
def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
              params=None, lr=None, trainer=None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat, y).sum()
            l.backward()
            if trainer is None:
                d2l.sgd(params, lr, batch_size)
            else:
                trainer.step(batch_size)
            y = y.astype('float32')
            train_l_sum += l.asscalar()
            train_acc_sum += (y_hat.argmax(axis=1) == y).sum().asscalar()
            n += y.size
            test_acc = evaluate_accuracy(test_iter, net)
            print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
                  % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))


# 训练
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
loss = cross_entropy
params = [W, b]
train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params, lr)

# 测试
for X, y in test_iter:
    break
true_labels = d2l.get_fashion_mnist_labels(y.asnumpy())
pred_labels = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1).asnumpy())
titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]  # zip是可迭代的对象
d2l.show_fashion_mnist(X[0:9], titles[0:9])
