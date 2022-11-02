from mxnet import nd

X = nd.arange(12).reshape((3, 4))
Y = nd.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print(X + Y)
print(X * Y)  # 对应元素乘
print(X / Y)  # 对应元素除
print(Y.exp())  # 指数运算 nd.exp(Y)
print(nd.dot(X, Y.T))  # 矩阵乘

print(nd.concat(X, Y, dim=0))  # 行连接
print(nd.concat(X, Y, dim=1))  # 列连接

print(X == Y)  # 每个位置都有比较结果

print(X.sum())  # 所有元素求和 nd.sum(X)

print(X.norm())  # 求L2范数 nd.norm(X)
print(X.norm().asscalar())  # 变为标量
