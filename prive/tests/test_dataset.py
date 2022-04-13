"""A  simple test for the dataset class"""
from unittest import TestCase

from warnings import filterwarnings

filterwarnings('ignore')

from prive.datasets import TabularDataset


class TestTabularDataset(TestCase):
    def test_read(self):
        data = TabularDataset.read('tests/data/texas')
        self.assertEqual(data.dataset.shape[0], 999)

    def test_sample(self):
        data = TabularDataset.read('tests/data/texas')

        # returns a subset of the samples
        data_sample = data.sample(500)

        self.assertEqual(data_sample.description, data.description)
        self.assertEqual(data_sample.dataset.shape[0], 500)

    def test_add(self):
        data = TabularDataset.read('tests/data/texas')

        # returns a subset of the samples
        data_sample1 = data.sample(500)
        data_sample2 = data.sample(500)

        data_1000 = data_sample1 + data_sample2

        self.assertEqual(data_1000.description, data_sample1.description)
        self.assertEqual(data_1000.dataset.shape[0], 1000)

    def test_get_records(self):
        data = TabularDataset.read('tests/data/texas')

        # returns a subset of the records
        index = [10, 20]
        record = data.get_records(index)

        self.assertEqual(record.dataset.iloc[0]['DISCHARGE'], data.dataset.iloc[index[0]]['DISCHARGE'])
        self.assertEqual(record.dataset.shape[0], len(index))

    def test_drop_records(self):
        data = TabularDataset.read('tests/data/texas')

        # returns a subset of the records
        index = [10, 20,50,100]
        new_dataset = data.drop_records(index)

        self.assertEqual(new_dataset.dataset.shape[0], data.dataset.shape[0]-len(index))

    def test_add_records(self):
        data = TabularDataset.read('tests/data/texas')

        # returns a subset of the records
        index = [100]
        record = data.get_records(index)

        new_dataset = data.add_records(record)

        self.assertEqual(new_dataset.dataset.shape[0], data.dataset.shape[0] + len(index))


if __name__ == '__main__':
    unittest.main()
