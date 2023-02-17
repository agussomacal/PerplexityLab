import unittest

import numpy as np
from src.Indexers import ArrayIndexerNd, EXTEND, CYCLIC


class TestIndexerNd(unittest.TestCase):
    def test_get_item(self):
        vec = np.arange(10).reshape((2, 5))
        indexer = ArrayIndexerNd(vec, modes=EXTEND)
        assert vec[indexer[(0, 0)]] == 0
        assert vec[indexer[(-1, -1)]] == 0
        assert vec[indexer[(1, -1)]] == 5
        assert vec[indexer[(2, 5)]] == 9
        assert np.all(vec[indexer[[(2, 5), (0, 0), (0, 0)]]] == [9, 0, 0])

        indexer = ArrayIndexerNd(vec, modes=CYCLIC)
        assert vec[indexer[(0, 0)]] == 0
        assert vec[indexer[(-1, -1)]] == 9
        assert vec[indexer[(1, -1)]] == 9
        assert vec[indexer[(2, 5)]] == 0
        assert np.all(vec[indexer[[(2, 5), (0, 0), (0, 0)]]] == [0, 0, 0])
        # assert np.all(indexer[[(1, 2), (2, 3), (3, 4)]] == [(1, 2), (0, 3), (1, 4)])
        assert isinstance(indexer[[(1, 2), (2, 3), (3, 4)]], tuple)
        assert np.all(indexer[[(1, 2), (2, 3), (3, 4)]][0] == np.array([1, 0, 1]))
        assert np.all(indexer[[(1, 2), (2, 3), (3, 4)]][1] == np.array([2, 3, 4]))

    def test_single_dimension(self):
        vec = np.arange(10).reshape((2, 5))
        indexer = ArrayIndexerNd(vec, modes=CYCLIC)
        assert np.all(indexer.transform_single_dimension([0, 1], dimension=0) == [0, 1])
        assert np.all(indexer.transform_single_dimension([0, 1, 2], dimension=0) == [0, 1, 0])
