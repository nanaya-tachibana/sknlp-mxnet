import os

from sknlp.vocab import Vocab
from sknlp.data import SimpleIndexedRecordIO
from sknlp.data import (
    SequenceTagDataset, ClassifyDataset, InMemoryDataset, NLPDataset,
    SupervisedNLPDataset, RecordFileDataset
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

    dataset_cls = NLPDataset

    def test_dataset(self, tmp_path):
        nlp_dataset = self.dataset_cls(self.dataset)
        assert len(nlp_dataset) == 3
        assert nlp_dataset[0] == [5, 7, 4]

    def test_custom_settings(self):
        nlp_dataset = self.dataset_cls(
            self.dataset,
            vocab=Vocab({
                '大': 102, '叫': 101, '好': 100,
                '家': 1, '厉': 1, '害': 1
            })
        )
        assert nlp_dataset[0] == [4, 5, 6]

    def test_max_length(self):
        nlp_dataset = self.dataset_cls(self.dataset, max_length=2)
        assert nlp_dataset[0] == [5, 7]

    def test_text_length(self):
        nlp_dataset = self.dataset_cls(self.dataset)
        assert nlp_dataset.text_lengths == [3, 3, 3]


class TestSupervisedNLPDataset(TestNLPDataset):

    dataset = InMemoryDataset(
        ['大叫好', '大家好', '好厉害'],
        ['1|2', '1|2|3', '3|1'],
        ['o1', 'o2', 'o3'],
        ['i1', 'i2', 'i3']
    )

    dataset_cls = SupervisedNLPDataset

    def test_dataset(self, tmp_path):
        dataset = self.dataset_cls(self.dataset)
        assert len(dataset) == 3
        assert dataset[0] == ([5, 7, 4], [0, 1])

    def test_custom_settings(self):
        dataset = self.dataset_cls(
            self.dataset,
            vocab=Vocab({
                '大': 102, '叫': 101, '好': 100,
                '家': 1, '厉': 1, '害': 1
            }),
            label2idx={'1': 2, '2': 0, '3': 1}
        )
        assert dataset[0] == ([4, 5, 6], [2, 0])

    def test_max_length(self):
        dataset = self.dataset_cls(self.dataset, max_length=2)
        assert dataset[0] == ([5, 7], [0, 1])


class TestClassifyDataset(TestSupervisedNLPDataset):

    dataset_cls = ClassifyDataset

    def test_dataset(self, tmp_path):
        dataset = self.dataset_cls(self.dataset)
        assert dataset[0] == ([5, 7, 4], [1, 1, 0])
        assert dataset.idx2labels([0, 8]) == ['1']

    def test_custom_settings(self):
        dataset = self.dataset_cls(
            self.dataset,
            vocab=Vocab({
                '大': 102, '叫': 101, '好': 100,
                '家': 1, '厉': 1, '害': 1
            }),
            label2idx={'1': 2, '2': 0, '3': 1}
        )
        assert dataset[0] == ([4, 5, 6], [1, 0, 1])

    def test_max_length(self):
        dataset = self.dataset_cls(self.dataset, max_length=2)
        assert dataset[0] == ([5, 7], [1, 1, 0])


class TestSequenceTagDataset(TestSupervisedNLPDataset):

    dataset = InMemoryDataset(
        ['大叫好', '大家好', '好厉害'],
        ['1|2|3', '1|4|3', '3|x|x'],
        ['o1', 'o2', 'o3'],
        ['i1', 'i2', 'i3']
    )

    dataset_cls = SequenceTagDataset

    def test_dataset(self, tmp_path):
        dataset = self.dataset_cls(self.dataset)
        assert dataset[0] == ([5, 7, 4], [0, 1, 2])
        assert dataset.idx2labels([0, 10]) == ['1', 'O']

    def test_custom_settings(self):
        dataset = self.dataset_cls(
            self.dataset,
            vocab=Vocab({
                '大': 102, '叫': 101, '好': 100,
                '家': 1, '厉': 1, '害': 1
            }),
            label2idx={'1': 2, '2': 0, '3': 1, '4': 3, 'x': 4}
        )
        assert dataset[0] == ([4, 5, 6], [2, 0, 1])

    def test_max_length(self):
        dataset = self.dataset_cls(self.dataset, max_length=2)
        assert dataset[0] == ([5, 7], [0, 1])
