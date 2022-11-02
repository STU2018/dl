from mxnet import nd

X = nd.arange(12).reshape((3, 4))
Y = nd.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

# 演示产生新内存
before = id(Y)
Y = Y + X
print(id(Y) == before)

# 创建相同形状的0矩阵
Z = Y.zeros_like()
print(Z)

# 全名函数out不产生新内存
before = id(Z)
nd.elemwise_add(X, Y, out=Z)
print(id(Z) == before)

# 利用索引机制不产生新内存
X[:] = X + Y
X += Y
