import os

import pytest

from sknlp.data.data import SimpleIndexedRecordIO


class TestSimpleIndexedRecordIO:

    IDX_FILE = 'tmp.idx'
    REC_FILE = 'tmp.rec'

    def test_read_write(self, tmp_path):
        idx_file = os.path.join(tmp_path, self.IDX_FILE)
        rec_file = os.path.join(tmp_path, self.REC_FILE)
        record = SimpleIndexedRecordIO(idx_file, rec_file, 'w')
        record.write(f'record_0,0|1'.encode('utf-8'))
        record.write(f'record_1,0|1'.encode('utf-8'))
        record.close()
        record = SimpleIndexedRecordIO(idx_file, rec_file, 'r')
        assert record.read_idx(1).decode('utf-8') == 'record_1,0|1'
