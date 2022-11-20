from mxnet import nd

A = nd.arange(3).reshape((3, 1))
B = nd.arange(2).reshape((1, 2))
print(A)
print(B)
print(A + B)
'''
形状不同的矩阵做运算，触发广播机制
广播：区域的复制
'''
