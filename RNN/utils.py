import random
import zipfile
from mxnet import autograd, nd
from mxnet.gluon import loss as gloss
import time
import math
import d2lzh as d2l
from mxnet.gluon import nn

# 加载数据
def load_data_jay_lyrics():
    with zipfile.ZipFile('../data/jaychou_lyrics.txt.zip') as zin:
        with zin.open('jaychou_lyrics.txt') as f:
            corpus_chars = f.read().decode('utf-8')
    corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')  # 把换行符换成空格
    corpus_chars = corpus_chars[0:10000]  # 只使用前10000个字符

    # 将每个字符映射成⼀个从0开始的连续整数
    idx_to_char = list(set(corpus_chars))  # set自动去重
    char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
    vocab_size = len(char_to_idx)

    corpus_indices = [char_to_idx[char] for char in corpus_chars]
    return corpus_indices, char_to_idx, idx_to_char, vocab_size


# 随机采样
def data_iter_random(corpus_indices, batch_size, num_steps, ctx=None):
    num_examples = (len(corpus_indices) - 1) // num_steps
    epoch_size = num_examples // batch_size  # 保证每个样本没有重合字符？
    example_indices = list(range(num_examples))  # 所有样本的起点
    random.shuffle(example_indices)

    # 返回从pos开始的⻓为num_steps的序列
    def _data(pos):
        return corpus_indices[pos: pos + num_steps]

    # 每次返回某个样本和他的标签（相邻的下一个样本）
    for i in range(epoch_size):
        i = i * batch_size
        batch_indices = example_indices[i: i + batch_size]  # batch_size 个起点
        X = [_data(j * num_steps) for j in batch_indices]
        Y = [_data(j * num_steps + 1) for j in batch_indices]
        yield nd.array(X, ctx), nd.array(Y, ctx)


# 连续采样
def data_iter_consecutive(corpus_indices, batch_size, num_steps, ctx=None):
    corpus_indices = nd.array(corpus_indices, ctx=ctx)
    data_len = len(corpus_indices)
    batch_len = data_len // batch_size
    indices = corpus_indices[0: batch_size * batch_len].reshape((batch_size, batch_len))  # 保证整除

    epoch_size = (batch_len - 1) // num_steps
    for i in range(epoch_size):
        i = i * num_steps
        X = indices[:, i: i + num_steps]
        Y = indices[:, i + 1: i + num_steps + 1]
        yield X, Y


# 生成one-hot向量
def to_onehot(X, size):
    return [nd.one_hot(x, size) for x in X.T]


# 获取初始化的隐藏状态
def init_rnn_state(batch_size, num_hiddens, ctx):
    # 使⽤元组是为了更便于处理隐藏状态含有多个NDArray的情况。
    return (nd.zeros(shape=(batch_size, num_hiddens), ctx=ctx),)


# 预测函数
def predict_rnn(prefix, num_chars, rnn, params, init_rnn_state,
                num_hiddens, vocab_size, ctx, idx_to_char, char_to_idx):
    state = init_rnn_state(1, num_hiddens, ctx)
    output = [char_to_idx[prefix[0]]]
    for t in range(num_chars + len(prefix) - 1):
        # 将上⼀时间步的输出作为当前时间步的输⼊
        X = to_onehot(nd.array([output[-1]], ctx=ctx), vocab_size)
        # 计算输出和更新隐藏状态
        (Y, state) = rnn(X, state, params)
        # 下⼀个时间步的输⼊是prefix⾥的字符或者当前的最佳预测字符
        if t < len(prefix) - 1:
            output.append(char_to_idx[prefix[t + 1]])
        else:
            output.append(int(Y[0].argmax(axis=1).asscalar()))
    return ''.join([idx_to_char[i] for i in output])


# 裁剪梯度,防止梯度爆炸，令梯度值不大于一个阈值theta
def grad_clipping(params, theta, ctx):
    norm = nd.array([0], ctx)
    for param in params:
        norm += (param.grad ** 2).sum()
    norm = norm.sqrt().asscalar()
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm


# 训练函数
def train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens,
                          vocab_size, ctx, corpus_indices, idx_to_char,
                          char_to_idx, is_random_iter, num_epochs, num_steps,
                          lr, clipping_theta, batch_size, pred_period,
                          pred_len, prefixes):
    if is_random_iter:
        data_iter_fn = data_iter_random
    else:
        data_iter_fn = data_iter_consecutive
    params = get_params()
    loss = gloss.SoftmaxCrossEntropyLoss()

    for epoch in range(num_epochs):
        if not is_random_iter:  # 如使⽤相邻采样，在epoch开始时初始化隐藏状态
            state = init_rnn_state(batch_size, num_hiddens, ctx)
        l_sum, n, start = 0.0, 0, time.time()
        data_iter = data_iter_fn(corpus_indices, batch_size, num_steps, ctx)

        for X, Y in data_iter:
            if is_random_iter:  # 如使⽤随机采样，在每个⼩批量更新前初始化隐藏状态
                state = init_rnn_state(batch_size, num_hiddens, ctx)
            else:  # 否则需要使⽤detach函数从计算图分离隐藏状态
                for s in state:
                    s.detach()  # 获取不需要梯度回传的部分
            with autograd.record():
                inputs = to_onehot(X, vocab_size)
                (outputs, state) = rnn(inputs, state, params)
                # outputs有num_steps个形状为(batch_size, vocab_size)的矩阵
                # 连结之后形状为(num_steps * batch_size, vocab_size)
                outputs = nd.concat(*outputs, dim=0)
                # Y的形状是(batch_size, num_steps)，转置后再变成⻓度为
                # batch * num_steps 的向量，这样跟输出的⾏⼀⼀对应
                y = Y.T.reshape((-1,))
                # 使⽤交叉熵损失计算平均分类误差
                l = loss(outputs, y).mean()
            l.backward()
            grad_clipping(params, clipping_theta, ctx)  # 裁剪梯度
            d2l.sgd(params, lr, 1)  # 因为误差已经取过均值，梯度不⽤再做平均(1)
            l_sum += l.asscalar() * y.size
            n += y.size
        if (epoch + 1) % pred_period == 0:
            print('epoch %d, perplexity %f, time %.2f sec' % (
                epoch + 1, math.exp(l_sum / n), time.time() - start))
            # 预测
            for prefix in prefixes:
                print(' -', predict_rnn(
                    prefix, pred_len, rnn, params, init_rnn_state,
                    num_hiddens, vocab_size, ctx, idx_to_char, char_to_idx))

class RNNModel(nn.Block):
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.dense = nn.Dense(vocab_size)

    def forward(self, inputs, state):
        X = nd.one_hot(inputs.T, self.vocab_size)
        Y, state = self.rnn(X, state)
        output = self.dense(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, *args, **kwargs):
        return self.rnn.begin_state(*args, **kwargs)