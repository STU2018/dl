from mxnet import autograd, nd
from utils import xyplot

x = nd.arange(-8.0, 8.0, 0.1)
x.attach_grad()
with autograd.record():
    y = x.tanh()
xyplot(x, y, 'tanh')

y.backward()
xyplot(x, x.grad, 'grad of tanh')