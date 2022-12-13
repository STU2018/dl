from mxnet import nd
import d2lzh as d2l

features, labels = d2l.get_data_ch7()


def init_adagrad_states():
    s_w = nd.zeros((features.shape[1], 1))
    s_b = nd.zeros(1)
    return (s_w, s_b)


def adagrad(params, states, hyperparams):
    eps = 1e-6
    for p, s in zip(params, states):
        s[:] += p.grad.square()
        p[:] -= hyperparams['lr'] * p.grad / (s + eps).sqrt()


d2l.train_ch7(adagrad, init_adagrad_states(), {'lr': 0.1}, features, labels)
