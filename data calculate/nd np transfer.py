from mxnet import nd
import numpy as np

P = np.ones((2, 3))
D = nd.array(P)  # np变nd
print(D)

print(D.asnumpy())  # nd变np
