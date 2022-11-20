from mxnet import nd

x = nd.arange(12)
print(x)

print(x.shape)
print(x.size)

X = x.reshape((3, 4))
print(X)

print(nd.zeros((2, 3, 4)))
print(nd.ones((3, 4)))

Y = nd.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])#用list构建array
print(Y)

# 均值为0，标准差为1
print(nd.random.normal(0, 1, shape=(3, 4)))
