import d2lzh as d2l
from mxnet import gluon, init
from mxnet.gluon import loss as gloss, nn

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

net = nn.Sequential()
net.add(nn.Dense(10))  # 输出为10的全连接层
net.initialize(init.Normal(sigma=0.01))  # 初始化参数

loss = gloss.SoftmaxCrossEntropyLoss()

trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})

num_epochs = 5
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None,
              None, trainer)
