import jieba_fast as jieba


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
            jieba.lcut('我们')
            self._method = jieba.lcut
        else:
            self._method = list

    def cut(self, text):
        return self._method(text)
