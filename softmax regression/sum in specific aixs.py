from mxnet import nd

X = nd.array([[1, 2, 3], [4, 5, 6]])
print(X.sum(axis=0, keepdims=True))
print(X.sum(axis=1, keepdims=True))

