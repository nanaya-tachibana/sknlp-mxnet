import jieba_fast as jieba
from pyhanlp import *
jieba.lcut('我们')
HanLP.segment('你好')


class Segmenter:
    """
    分词器调用接口

    Parameters
    ----------
    method: `str`
        分词器名, 可选项: 'jieba', `None`, 如果是None
    """

    def __init__(self, method: str = None) -> None:
        if method == 'jieba':
            self._method = jieba.lcut
        elif method == 'hanlp':
            self._method = lambda x: [s.word for s in HanLP.segment(x)]
        elif method == 'space':
            self._method = lambda x: x.split()
        else:
            self._method = list

    def cut(self, text):
        return self._method(text)
