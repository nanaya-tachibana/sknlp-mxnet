import os

from multiprocessing import current_process
import ctypes

import mxnet as mx
from mxnet.gluon.data.dataset import Dataset
import pandas as pd

from mxnet.base import _LIB

from mxnet.base import check_call


class MXIndexedRecordIO(mx.recordio.MXRecordIO):
    """Reads/writes `RecordIO` data format, supporting random access.

    Examples
    ---------
    >>> for i in range(5):
    ...     record.write_idx(i, 'record_%d'%i)
    >>> record.close()
    >>> record = mx.recordio.MXIndexedRecordIO('tmp.idx', 'tmp.rec', 'r')
    >>> record.read_idx(3)
    record_3

    Parameters
    ----------
    idx_path : str
        Path to the index file.
    uri : str
        Path to the record file. Only supports seekable file types.
    flag : str
        'w' for write or 'r' for read.
    key_type : type
        Data type for keys.
    """

    def __init__(self, idx_path, uri, flag, key_type=int):
        self.idx_path = idx_path
        self.fidx = None
        self.pid = None
        super(MXIndexedRecordIO, self).__init__(uri, flag)

    def open(self):
        super(MXIndexedRecordIO, self).open()
        self.pid = current_process().pid
        self.fidx = open(self.idx_path, self.flag)
        if not self.writable:
            df = pd.read_csv(self.idx_path, header=None)
            self.positions = df[0].values

    def close(self):
        """Closes the record file."""
        if not self.is_open:
            return
        super(MXIndexedRecordIO, self).close()
        self.fidx.close()
        self.pid = None

    def __getstate__(self):
        """Override pickling behavior."""
        d = super(MXIndexedRecordIO, self).__getstate__()
        d['fidx'] = None
        return d

    def seek(self, idx):
        """Sets the current read pointer position.

        This function is internally called by `read_idx(idx)` to find the current
        reader pointer position. It doesn't return anything."""
        assert not self.writable
        self._check_pid(allow_reset=True)
        pos = ctypes.c_size_t(self.positions[idx])
        check_call(_LIB.MXRecordIOReaderSeek(self.handle, pos))

    def tell(self):
        """Returns the current position of write head.

        Examples
        ---------
        >>> record = mx.recordio.MXIndexedRecordIO('tmp.idx', 'tmp.rec', 'w')
        >>> print(record.tell())
        0
        >>> for i in range(5):
        ...     record.write_idx(i, 'record_%d'%i)
        ...     print(record.tell())
        16
        32
        48
        64
        80
        """
        assert self.writable
        pos = ctypes.c_size_t()
        check_call(_LIB.MXRecordIOWriterTell(self.handle, ctypes.byref(pos)))
        return pos.value

    def read_idx(self, idx):
        """Returns the record at given index.

        Examples
        ---------
        >>> record = mx.recordio.MXIndexedRecordIO('tmp.idx', 'tmp.rec', 'w')
        >>> for i in range(5):
        ...     record.write_idx(i, 'record_%d'%i)
        >>> record.close()
        >>> record = mx.recordio.MXIndexedRecordIO('tmp.idx', 'tmp.rec', 'r')
        >>> record.read_idx(3)
        record_3
        """
        self.seek(idx)
        return self.read()

    def write(self, buf):
        """Inserts input record at given index.

        Examples
        ---------
        >>> for i in range(5):
        ...     record.write_idx(i, 'record_%d'%i)
        >>> record.close()

        Parameters
        ----------
        idx : int
            Index of a file.
        buf :
            Record to write.
        """
        pos = self.tell()
        super().write(buf)
        self.fidx.write('%d\n' % pos)

    def _check_pid(self, allow_reset=False):
        """Check process id to ensure integrity, reset if in new process."""
        if not self.pid == current_process().pid:
            if allow_reset:
                self.reset()
            else:
                raise RuntimeError("Forbidden operation in multiple processes")


class RecordFileDataset(Dataset):
    """A dataset wrapping over a RecordIO (.rec) file.

    Each sample is a string representing the raw content of an record.

    Parameters
    ----------
    filename : str
        Path to rec file.
    """

    def __init__(self, filename):
        self.idx_file = os.path.splitext(filename)[0] + '.idx'
        self.filename = filename
        self._record = MXIndexedRecordIO(self.idx_file, self.filename, 'r')

    def __getitem__(self, idx):
        return self._record.read_idx(idx)

    def __len__(self):
        return len(self._record.positions)
