from mxnet import autograd, nd
from mxnet import init
from mxnet import gluon
from mxnet.gluon import loss as gloss
from mxnet.gluon import nn
from mxnet.gluon import data as gdata

# 生成数据集
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = nd.random.normal(scale=1, shape=(num_examples, num_inputs))
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += nd.random.normal(scale=0.01, shape=labels.shape)

batch_size = 10

# 将训练数据的特征和标签组合
dataset = gdata.ArrayDataset(features, labels)

# 随机读取⼩批量
data_iter = gdata.DataLoader(dataset, batch_size, shuffle=True)

# 定义模型
net = nn.Sequential()
net.add(nn.Dense(1))  # 该层输出个数为1

# 初始化模型参数 标准差为0.01
net.initialize(init.Normal(sigma=0.01))

# 定义损失函数
loss = gloss.L2Loss()  # 平⽅损失

# 定义优化算法
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.03})

# 模型训练
num_epochs = 3
for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        with autograd.record():
            l = loss(net(X), y)
        l.backward()
        trainer.step(batch_size)
    l = loss(net(features), labels)
    print('epoch %d, loss: %f' % (epoch, l.mean().asnumpy()))
