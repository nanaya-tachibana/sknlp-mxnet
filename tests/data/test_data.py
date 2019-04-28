import os
import pytest
from sknlp.data.data import (SimpleIndexedRecordIO, RecordFileDataset,
                             InMemoryDataset)


IDX_FILE = 'tmp.idx'
REC_FILE = 'tmp.rec'


def set_up(tmp_path):
    record = SimpleIndexedRecordIO(os.path.join(tmp_path, IDX_FILE),
                                   os.path.join(tmp_path, REC_FILE),
                                   'w')
    for i in range(5):
        d = f'record_{i},{"|".join([str(c) for c in range(i)])}'
        record.write(d.encode('utf-8'))
    record.close()


class TestSimpleIndexedRecordIO:

    def test_read_write(self, tmp_path):
        set_up(tmp_path)
        record = SimpleIndexedRecordIO(os.path.join(tmp_path, IDX_FILE),
                                       os.path.join(tmp_path, REC_FILE),
                                       'r')
        assert record.read_idx(2).decode('utf-8') == 'record_2,0|1'


class TestRecordFileDataset:

    IDX_FILE = 'tmp.idx'
    REC_FILE = 'tmp.rec'

    def test_read_write(self, tmp_path):
        set_up(tmp_path)
        dataset = RecordFileDataset(os.path.join(tmp_path, REC_FILE))
        assert dataset[2].decode('utf-8') == 'record_2,0|1'
