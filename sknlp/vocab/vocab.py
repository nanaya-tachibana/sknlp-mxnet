import numpy as np
import gluonnlp
from gensim.models import KeyedVectors
from typing import Tuple


class Vocab(gluonnlp.Vocab):

    @classmethod
    def from_word2vec_file(cls,
                           file_path: str,
                           binary: bool = True) -> Tuple[Vocab, np.ndarray]:
        """
        从word2vec格式文件读取词汇表

        Parameters
        ----------
        file_path: `str`
            word2vec文件路径
        binary: `bool`, optional(default=`True`)
            word2vec文件是否以二进制存储

        Returns
        ----------
        vocab: `Vocab`
            词汇表
        embed: `numpy.array`, shape(vocab_size, embed_size)
            embedding矩阵
        """
        embed = KeyedVectors.load_word2vec_format(file_path, binary=binary)
        counter = gluonnlp.data.count_tokens(embed.vocab.keys())
        vocab = gluonnlp.Vocab(counter)
        embed_weight = embed[vocab.idx_to_token[4:]]
        embed_weight = np.concatenate(np.zeros((4, embed_weight.shape[1])),
                                      embed_weight, axis=0)
        return vocab, embed_weight
