from mxnet import nd, autograd

x = nd.arange(4).reshape((4, 1))
print(x)
x.attach_grad()  # 申请储存梯度需要的内存
with autograd.record():  # 要求记录与梯度有关的计算
    y = 2 * nd.dot(x.T, x)
y.backward()  # 自动求梯度
print(x.grad)    #grad 其实是求和
assert (x.grad - 4 * x).norm().asscalar() == 0

print(autograd.is_training())
with autograd.record():  # 此时会从预测模式转为训练模式
    print(autograd.is_training())


#体会求和
x = nd.arange(8).reshape((4, 2))
print(x)
x.attach_grad()  # 申请储存梯度需要的内存
with autograd.record():  # 要求记录与梯度有关的计算
    y = 2 * nd.dot(x.T, x)
y.backward()  # 自动求梯度
print(x.grad)    #grad 其实是求和
