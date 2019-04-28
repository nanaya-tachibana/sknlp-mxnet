import os
from typing import IO, List, Dict, Tuple

from multiprocessing import current_process
import collections
import ctypes

import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

import mxnet as mx
from mxnet.base import _LIB
from mxnet.base import check_call
from mxnet.gluon.data.dataset import Dataset

import gluonnlp

from .segmener import Segmenter


class SimpleIndexedRecordIO(mx.recordio.MXIndexedRecordIO):
    """
    Indexed ``RecordIO`` data format, supporting random access.

    Examples
    ---------
    record = SimpleIndexedRecordIO('tmp.idx', 'tmp.rec', 'w')
    >>> for i in range(5):
    ...    d = f'record_{i}'
    ...    record.write(d.encode('utf-8'))
    >>> record.close()
    >>> record = SimpleIndexedRecordIO('tmp.idx', 'tmp.rec', 'r')
    >>> record.read_idx(3)
    record_3

    Parameters
    ----------
    idx_path: ``str``
        Path to the index file of ``RecordIO`` file.
    uri: ``str``
        Path to ``RecordIO`` file.
    flag: ``str``
        'w' for write or 'r' for read.
    """

    def __init__(self, idx_path: str, uri: str, flag: str) -> None:
        self.positions = np.array([], dtype=np.uint)
        super().__init__(idx_path, uri, flag)

    def open(self) -> None:
        """
        打开``RecordIO``文件
        """
        if self.flag == "w":
            check_call(_LIB.MXRecordIOWriterCreate(self.uri,
                                                   ctypes.byref(self.handle)))
            self.writable = True
        elif self.flag == "r":
            check_call(_LIB.MXRecordIOReaderCreate(self.uri,
                                                   ctypes.byref(self.handle)))
            self.writable = False
        else:
            raise ValueError("Invalid flag %s" % self.flag)
        self.pid = current_process().pid
        self.is_open = True
        self.fidx: IO = open(self.idx_path, self.flag)  # 兼容父类close方法
        if not self.writable:
            self.positions = pd.read_csv(self.idx_path,
                                         header=None, dtype=np.uint)[0].values

    def seek(self, idx: int) -> None:
        """
        Sets the current read pointer position.

        This function is internally called by `read_idx(idx)` to
        find the current reader pointer position.
        It doesn't return anything.
        """
        assert not self.writable
        self._check_pid(allow_reset=True)
        pos = ctypes.c_size_t(self.positions[idx])
        check_call(_LIB.MXRecordIOReaderSeek(self.handle, pos))

    def write(self, buf: bytes) -> None:
        """
        Write ``buf`` sequentially.

        Examples
        ---------
        >>> for i in range(5):
        ...     record.write(f'record_{i}'.encode('utf-8'))
        >>> record.close()

        Parameters
        ----------
        buf: ``byte``
            Record to write.
        """
        pos = self.tell()
        super().write(buf)
        self.fidx.write(f'{pos}\n')


class RecordFileDataset(Dataset):
    """
    A dataset wrapper for a ``SimpleIndexedRecordIO`` file.

    Each sample is a string representing the raw content of an record.

    Parameters
    ----------
    filename : ``str``
        Path to ``RecordIO`` file.
    """

    def __init__(self, filename: str, encode: str = 'utf-8') -> None:
        self.idx_file = os.path.splitext(filename)[0] + '.idx'
        self.filename = filename
        self._encode = encode
        self._record = SimpleIndexedRecordIO(self.idx_file, self.filename, 'r')

    def __getitem__(self, idx: int) -> str:
        return self._record.read_idx(idx).decode('utf-8')

    def __len__(self) -> int:
        return len(self._record.positions)


class InMemoryDataset(Dataset):
    """
    A dataset wrapper for lists.
    """

    def __init__(self, *args: List[List[str]]) -> None:
        self._record = ['\t'.join(tuples) for tuples in zip(*args)]

    def __getitem__(self, idx: int) -> str:
        return self._record[idx]

    def __len__(self) -> int:
        return len(self._record)


class NLPDatasetMixin:
    """
    Mixin class实现了基本的NLP预处理, 来预处理``Dataset``.

    Parameters
    ----------
    vocab: gluonnlp.Vocab, optional
        词汇表, 如果为None, 会根据数据集构建

    """

    def __init__(self, vocab: gluonnlp.Vocab = None,
                 label2idx: Dict[str, int] = None,
                 segmenter: str = None,
                 max_length: int = 100,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        cutter = Segmenter(segmenter)
        self._segmenter = cutter.cut
        self._max_length = max_length

        if label2idx is None or vocab is None:
            counter = collections.Counter()
            label_set = set()
            for i in range(super().__len__()):
                row = super().__getitem__(i)
                text, label = row.split('\t')
                if vocab is None:
                    counter.update(self._segmenter(text))
                if label2idx is None:
                    labels = label.split('|')
                    label_set.update(labels)
        if vocab is None:
            self.vocab = gluonnlp.Vocab(counter)
        else:
            self.vocab = vocab
        if label2idx is None:
            label_set.discard('')
            label_list = list(label_set)
            label_list.sort()
            self.label2idx = dict(zip(label_list, range(len(label_set))))
        else:
            self.label2idx = label2idx
        self._idx2label = {v: k for k, v in self.label2idx.items()}

    def _preprocess_text(self, text: str) -> List[int]:
        return self.vocab[self._segmenter(text[:self._max_length])]

    def _preprocess_label(self, label: str) -> List[int]:
        return [self.label2idx[l] for l in label.split('|')]

    def _preprocess_func(self, row: str) -> Tuple[List[int],
                                                  List[int],
                                                  List[int]]:
        text, label = row.split('\t')
        processed_text = self._preprocess_text(text)
        processed_label = self._preprocess_label(label)
        mask = [1] * len(text)
        return processed_text, mask, processed_label

    def idx2tokens(self, idx_list):
        return self.vocab.to_tokens(idx_list)

    def __getitem__(self, idx):
        row = super().__getitem__(idx)
        return self._preprocess_func(row)


class ClassifyDatasetMixin(NLPDatasetMixin):

    def __init__(self, vocab=None, label2idx=None, segmenter=None,
                 encode='utf-8', max_length=100, **kwargs):
        super().__init__(vocab=vocab, label2idx=label2idx,
                         segmenter=segmenter, encode=encode,
                         max_length=max_length, **kwargs)
        self._binarizer = MultiLabelBinarizer([
            self._idx2label[i] for i in range(len(self.label2idx))])

    def _preprocess_label(self, label):
        return self._binarizer.fit_transform([label.split('|')])[0]\
                              .astype(np.float32)

    def idx2labels(self, idx_list):
        return [self._idx2label.get(i, None) for i in idx_list]


class SequenceTagDatasetMixin(NLPDatasetMixin):

    def _preprocess_label(self, label):
        return [self.label2idx[l] for l in label.split('|')[:self._max_length]]

    def idx2labels(self, idx_list):
        return [self._idx2label.get(i, 'O') for i in idx_list]


class _SimpleClassifyDataset(ClassifyDatasetMixin, InMemoryDataset):
    """
    A dataset wrapper for `InMemoryDataset` with `NLPDatasetMixin`.

    Examples
    ---------
    >>> ds = _SimpleClassifyDataset(['大叫好', '大家好', '好厉害'], ['1|2', '1|2|3', '3|1'])
    >>> len(ds)
    3
    >>> ds[0]
    ([5, 7, 4], [2, 1])
    """

    def __init__(self, text_list, label_list, vocab=None, label2idx=None,
                 segmenter=None, encode=None, max_length=100):
        super().__init__(text_list=text_list,
                         label_list=label_list,
                         vocab=vocab,
                         label2idx=label2idx,
                         segmenter=segmenter,
                         encode=encode,
                         max_length=max_length)


class _SimpleSequenceTagDataset(SequenceTagDatasetMixin, InMemoryDataset):

    def __init__(self, text_list, label_list, vocab=None, label2idx=None,
                 segmenter=None, encode=None, max_length=100):
        super().__init__(text_list=text_list,
                         label_list=label_list,
                         vocab=vocab,
                         label2idx=label2idx,
                         segmenter=segmenter,
                         encode=encode,
                         max_length=max_length)


DATASET_DIR = 'datasets'


class MsraDataset(SequenceTagDatasetMixin, RecordFileDataset):

    DIR = 'msra'

    def __init__(self, is_train_file=True, vocab=None, label2idx=None,
                 segmenter=None, encode='utf-8', max_length=100):
        filename = 'train.rec' if is_train_file else 'test.rec'
        super().__init__(filename=os.path.join(DATASET_DIR, self.DIR,
                                               filename),
                         vocab=vocab,
                         label2idx=label2idx,
                         segmenter=segmenter,
                         encode=encode,
                         max_length=max_length)


class WaimaiDataset(ClassifyDatasetMixin, RecordFileDataset):

    DIR = 'waimai'

    def __init__(self, is_train_file=True, vocab=None, label2idx=None,
                 segmenter=None, encode='utf-8', max_length=100):
        filename = 'train.rec' if is_train_file else 'test.rec'
        super().__init__(filename=os.path.join(DATASET_DIR, self.DIR,
                                               filename),
                         vocab=vocab,
                         label2idx=label2idx,
                         segmenter=segmenter,
                         encode=encode,
                         max_length=max_length)


class IntentDataset(ClassifyDatasetMixin, RecordFileDataset):

    DIR = 'intent'

    def __init__(self, is_train_file=True, vocab=None, label2idx=None,
                 segmenter=None, encode='utf-8', max_length=100):
        filename = 'train.rec' if is_train_file else 'test.rec'
        super().__init__(filename=os.path.join(DATASET_DIR, self.DIR,
                                               filename),
                         vocab=vocab,
                         label2idx=label2idx,
                         segmenter=segmenter,
                         encode=encode,
                         max_length=max_length)

    def _preprocess_label(self, label):
        if label == 'nonsense':
            label = ''
        return self._binarizer.fit_transform([label.split('|')])[0]\
                              .astype(np.float32)


def load_dataset(dataset, segmenter='jieba'):
    train_dataset = dataset(True, segmenter=segmenter)
    test_dataset = dataset(False,
                           vocab=train_dataset.vocab,
                           label2idx=train_dataset.label2idx,
                           segmenter=segmenter)
    return train_dataset, test_dataset


def load_msra_dataset(segmenter=None):
    return load_dataset(MsraDataset, segmenter=segmenter)


def load_waimai_dataset(segmenter='jieba'):
    return load_dataset(WaimaiDataset, segmenter=segmenter)


def load_intent_dataset(segmenter='jieba'):
    return load_dataset(IntentDataset, segmenter=segmenter)
