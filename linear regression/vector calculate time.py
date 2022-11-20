from mxnet import nd
from time import time

a = nd.ones(shape=1000)
# 生成的是行向量，不同于a=nd.ones((1,1000))
b = nd.ones(shape=1000)

start = time()
c = nd.zeros(shape=1000)

# 逐个相加
for i in range(1000):
    c[i] = a[i] + b[i]
print("耗时：{0:.9f}".format(time() - start))

# 矢量相加
start = time()
d = a + b
print("耗时：{0:.9f}".format(time() - start))
