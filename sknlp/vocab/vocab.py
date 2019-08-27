from collections import Counter
from typing import Tuple

import numpy as np
import gluonnlp
from gensim.models import KeyedVectors


class Vocab(gluonnlp.Vocab):

    @classmethod
    def from_word2vec_file(
            cls, file_path: str, binary: bool = True
    ) -> Tuple['Vocab', np.ndarray]:
        """
        从word2vec格式文件读取词汇表

        Parameters
        ----------
        file_path: word2vec文件路径
        binary: word2vec文件是否以二进制存储, default=True

        Returns
        ----------
        vocab: 词汇表
        embed: embedding矩阵, shape(vocab_size, embed_size)
        """
        embed = KeyedVectors.load_word2vec_format(file_path, binary=binary)
        tokens = Counter(embed.vocab.keys())
        vocab = cls(tokens)
        embed_weight = embed[vocab.idx_to_token[4:]]
        d = embed_weight.shape[1]
        embed_weight = np.concatenate([
            embed[['<unk>']] if '<unk>' in embed else np.zeros((1, d)),
            embed[['<pad>']] if '<pad>' in embed else np.zeros((1, d)),
            embed[['<bos>']] if '<bos>' in embed else np.zeros((1, d)),
            embed[['<eos>']] if '<eos>' in embed else np.zeros((1, d)),
            embed_weight], axis=0)
        return vocab, embed_weight
