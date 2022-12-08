from mxnet import nd
import random
import zipfile


def load_data_jay_lyrics():
    with zipfile.ZipFile('../data/jaychou_lyrics.txt.zip') as zin:
        with zin.open('jaychou_lyrics.txt') as f:
            corpus_chars = f.read().decode('utf-8')
    corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
    corpus_chars = corpus_chars[0:10000]  # 只使用前10000个字符

    # 将每个字符映射成⼀个从0开始的连续整数
    idx_to_char = list(set(corpus_chars))  # set自动去重
    char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
    vocab_size = len(char_to_idx)

    corpus_indices = [char_to_idx[char] for char in corpus_chars]
    sample = corpus_indices[:20]
    return corpus_indices, char_to_idx, idx_to_char, vocab_size


def data_iter_random(corpus_indices, batch_size, num_steps, ctx=None):
    num_examples = (len(corpus_indices) - 1) // num_steps
    epoch_size = num_examples // batch_size
    example_indices = list(range(num_examples))  # 所有样本的起点
    random.shuffle(example_indices)

    # 返回从pos开始的⻓为num_steps的序列
    def _data(pos):
        return corpus_indices[pos: pos + num_steps]

    # 每次返回某个样本和他的标签（相邻的下一个样本）
    for i in range(epoch_size):
        i = i * batch_size
        batch_indices = example_indices[i: i + batch_size]
        X = [_data(j * num_steps) for j in batch_indices]
        Y = [_data(j * num_steps + 1) for j in batch_indices]
        yield nd.array(X, ctx), nd.array(Y, ctx)
