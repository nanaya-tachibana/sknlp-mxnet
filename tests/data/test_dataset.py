import os

from sknlp.vocab import Vocab
from sknlp.data import SimpleIndexedRecordIO
from sknlp.data import (
    ClassifyDataset, InMemoryDataset, NLPDataset,
    RecordFileDataset, SequenceTagDataset
)


class TestDataset:

    IDX_FILE = 'tmp.idx'
    REC_FILE = 'tmp.rec'

    def set_up(self, tmp_path):
        record = SimpleIndexedRecordIO(
            os.path.join(tmp_path, self.IDX_FILE),
            os.path.join(tmp_path, self.REC_FILE), 'w'
        )
        for i in range(5):
            d = f'record_{i}\t{"|".join([str(c) for c in range(i)])}'
            record.write(d.encode('utf-8'))
        record.close()


class TestRecordFileDataset(TestDataset):

    def test_dataset(self, tmp_path):
        self.set_up(tmp_path)
        dataset = RecordFileDataset(os.path.join(tmp_path, self.REC_FILE))
        assert len(dataset) == 5
        assert dataset[2] == 'record_2\t0|1'


class TestInMemoryDataset(TestDataset):

    def test_dataset(self, tmp_path):
        dataset = InMemoryDataset(
            ['record_0', 'record_1', 'record_2'],
            ['0|1', '1|2', '2|3'],
            ['o1', 'o2', 'o3'],
            ['i1', 'i2', 'i3']
        )
        assert len(dataset) == 3
        assert dataset[2] == 'record_2\t2|3\to3\ti3'


class TestNLPDataset(TestDataset):

    dataset = InMemoryDataset(
        ['大叫好', '大家好', '好厉害'],
        ['1|2', '1|2|3', '3|1'],
        ['o1', 'o2', 'o3'],
        ['i1', 'i2', 'i3']
    )

    def test_dataset(self, tmp_path):
        nlp_dataset = NLPDataset(self.dataset)
        assert len(nlp_dataset) == 3
        assert nlp_dataset[0] == ([5, 7, 4], [0, 1], ['o1', 'i1'])

    def test_custom_settings(self):
        nlp_dataset = NLPDataset(
            self.dataset,
            vocab=Vocab({
                '大': 102, '叫': 101, '好': 100,
                '家': 1, '厉': 1, '害': 1
            }),
            label2idx={'1': 2, '2': 0, '3': 1}
        )
        assert nlp_dataset[0] == ([4, 5, 6], [2, 0], ['o1', 'i1'])

    def test_max_length(self):
        nlp_dataset = NLPDataset(self.dataset, max_length=2)
        assert nlp_dataset[0] == ([5, 7], [0, 1], ['o1', 'i1'])


class TestClassifyDataset(TestNLPDataset):

    def test_dataset(self, tmp_path):
        c_dataset = ClassifyDataset(self.dataset)
        assert c_dataset[0] == ([5, 7, 4], [1, 1, 0], ['o1', 'i1'])
        assert c_dataset.idx2labels([0, 8]) == ['1']


class TestSequenceTagDataset(TestDataset):

    dataset = InMemoryDataset(
        ['大叫好', '大家好', '好厉害'],
        ['1|2|3', '1|4|3', '3|x|x'],
        ['o1', 'o2', 'o3'],
        ['i1', 'i2', 'i3']
    )

    def test_dataset(self, tmp_path):
        s_dataset = SequenceTagDataset(self.dataset)
        assert s_dataset[0] == ([5, 7, 4], [0, 1, 2], ['o1', 'i1'])
        assert s_dataset.idx2labels([0, 10]) == ['1', 'O']
