import ctypes
from multiprocessing import current_process
from typing import IO

import mxnet as mx
import numpy as np
import pandas as pd
from mxnet.base import _LIB, check_call


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
            check_call(_LIB.MXRecordIOWriterCreate(
                self.uri, ctypes.byref(self.handle)))
            self.writable = True
        elif self.flag == "r":
            check_call(_LIB.MXRecordIOReaderCreate(
                self.uri, ctypes.byref(self.handle)))
            self.writable = False
        else:
            raise ValueError("Invalid flag %s" % self.flag)
        self.pid = current_process().pid
        self.is_open = True
        self.fidx: IO = open(self.idx_path, self.flag)  # 兼容父类close方法
        if not self.writable:
            self.positions = pd.read_csv(
                self.idx_path, header=None, dtype=np.uint
            )[0].values

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
