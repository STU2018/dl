from mxnet import nd

X = nd.arange(12).reshape((3, 4))
print(X[1:3])#第一维切片

X[1, 2] = 9
print(X)

X[1:2, :] = 12
print(X)
