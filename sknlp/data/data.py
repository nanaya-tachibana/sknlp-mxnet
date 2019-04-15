import os
from typing import IO, Optional

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

from .utils import word_cut_func


class SimpleIndexedRecordIO(mx.recordio.MXIndexedRecordIO):
    """
    Indexed ``RecordIO`` data format, 支持随机存取.

    Examples
    ---------
    record = SimpleIndexedRecordIO('tmp.idx', 'tmp.rec', 'w')
    >>> for i in range(5):
    ...    d = 'record_%d' % i
    ...    record.write(d.encode('utf-8'))
    >>> record.close()
    >>> record = SimpleIndexedRecordIO('tmp.idx', 'tmp.rec', 'r')
    >>> record.read_idx(3)
    record_3

    Parameters
    ----------
    idx_path: ``str``
        index文件路径
    uri: ``str``
        record文件路径, 仅支持seekable文件类型
    flag: ``str``
        'w'(写)或者'r'(读)
    """

    def __init__(self, idx_path: str, uri: str, flag: str) -> None:
        self.idx_path = idx_path
        self.flag = flag
        super().__init__(idx_path, uri, flag)
        self.fidx: IO = open(self.idx_path, self.flag)  # 兼容父类close方法

    def open(self) -> None:
        """

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
        Inserts input record.

        Examples
        ---------
        >>> for i in range(5):
        ...     record.write('record_%d'%i)
        >>> record.close()

        Parameters
        ----------
        idx : int
            Index of a file.
        buf : byte
            Record to write.
        """
        pos = self.tell()
        super().write(buf)
        self.fidx.write('%d\n' % pos)


class RecordFileDataset(Dataset):
    """
    A dataset wrapper for a RecordIO (.rec) file.

    Each sample is a string representing the raw content of an record.

    Parameters
    ----------
    filename : str
        Path to rec file.
    """

    def __init__(self, filename):
        self.idx_file = os.path.splitext(filename)[0] + '.idx'
        self.filename = filename
        self._record = SimpleIndexedRecordIO(self.idx_file, self.filename, 'r')

    def __getitem__(self, idx):
        return self._record.read_idx(idx)

    def __len__(self):
        return len(self._record.positions)


class InMemoryDataset(Dataset):
    """
    A dataset wrapper for lists.


    Parameters
    ----------
    text_list: list
        List of text.
    label_list: list
        List of label
    """

    def __init__(self, text_list, label_list):
        self._record = ['\t'.join([text, label])
                        for text, label in zip(text_list, label_list)]

    def __getitem__(self, idx):
        return self._record[idx]

    def __len__(self):
        return len(self._record)


class NLPDatasetMixin:
    """
    A dataset mixin which implements basic preprocess for NLP data.

    Always used with a `Dataset`.
    """

    def __init__(self, vocab=None, label2idx=None, segmenter=list,
                 encode='utf-8', max_length=100, **kwargs):
        super().__init__(**kwargs)
        self.vocab = vocab
        self.label2idx = label2idx
        self._segmenter = segmenter
        self._encode = encode
        self._max_length = max_length

        if self.label2idx is None or self.vocab is None:
            counter = collections.Counter()
            label_set = set()
            for i in range(super().__len__()):
                row = super().__getitem__(i)
                if self._encode is not None:
                    row = row.decode(self._encode)
                text, label = row.split('\t')
                if self.vocab is None:
                    counter.update(self._segmenter(text))
                if self.label2idx is None:
                    labels = label.split('|')
                    label_set.update(labels)
        if self.vocab is None:
            self.vocab = gluonnlp.Vocab(counter)
        if self.label2idx is None:
            label_set.discard('')
            label_list = list(label_set)
            label_list.sort()
            self.label2idx = dict(zip(label_list, range(len(label_set))))
        self._idx2label = {v: k for k, v in self.label2idx.items()}

    def _preprocess_text(self, text):
        return self.vocab[self._segmenter(text[:self._max_length])]

    def _preprocess_label(self, label):
        return [self.label2idx[l] for l in label.split('|')]

    def _preprocess_func(self, row):
        if self._encode is not None:
            row = row.decode('utf-8')
        text, label = row.split('\t')
        text = self._preprocess_text(text)
        label = self._preprocess_label(label)
        mask = np.ones(len(text), dtype=np.float32)
        return text, mask, label

    def idx2tokens(self, idx_list):
        return self.vocab.to_tokens(idx_list)

    def __getitem__(self, idx):
        row = super().__getitem__(idx)
        return self._preprocess_func(row)


class ClassifyDatasetMixin(NLPDatasetMixin):

    def __init__(self, vocab=None, label2idx=None, segmenter=list,
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
                 segmenter=list, encode=None, max_length=100):
        super().__init__(text_list=text_list,
                         label_list=label_list,
                         vocab=vocab,
                         label2idx=label2idx,
                         segmenter=segmenter,
                         encode=encode,
                         max_length=max_length)


class _SimpleSequenceTagDataset(_SimpleClassifyDataset,
                                SequenceTagDatasetMixin):
    pass


DATASET_DIR = 'datasets'


class MsraDataset(SequenceTagDatasetMixin, RecordFileDataset):

    DIR = 'msra'

    def __init__(self, is_train_file=True, vocab=None, label2idx=None,
                 segmenter=list, encode='utf-8', max_length=100):
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
                 segmenter=list, encode='utf-8', max_length=100):
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
                 segmenter=list, encode='utf-8', max_length=100):
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


def load_dataset(dataset):
    train_dataset = dataset(True, segmenter=word_cut_func)
    test_dataset = dataset(False,
                           vocab=train_dataset.vocab,
                           label2idx=train_dataset.label2idx,
                           segmenter=word_cut_func)
    return train_dataset, test_dataset


def load_msra_dataset():
    return load_dataset(MsraDataset)


def load_waimai_dataset():
    return load_dataset(WaimaiDataset)


def load_intent_dataset():
    return load_dataset(IntentDataset)
