import itertools
from numbers import Integral
from typing import List, Iterable, Tuple, Union

import numpy as np

CYCLIC = "cyclic"
EXTEND = "extend"
REFLECT = "reflect"


def get_cyclic_indexes(array, indexes, cyclic_dimension):
    max_dim_size = np.shape(array)[cyclic_dimension]
    if isinstance(indexes, int):
        indexes = [indexes]
    elif isinstance(indexes, slice):
        step = 1 if indexes.step is None else indexes.step
        start = 0 if indexes.start is None else indexes.start % max_dim_size
        stop = max_dim_size if indexes.stop is None else indexes.stop % max_dim_size
        indexes = list(range(start, stop, step))

    return np.array([i % max_dim_size for i in indexes])


class ArrayIndexer:
    def __init__(self, array_len, mode=EXTEND):
        self.array_len = array_len
        self.mode = mode.lower()
        self.transformer_func = getattr(self, 'transform_indexes2{}_indexes'.format(self.mode))

        assert self.mode == EXTEND or self.mode == CYCLIC, 'mode should be extend or cyclic'

    def __len__(self):
        return self.array_len

    def __getitem__(self, item):
        if isinstance(item, Integral):
            return self.transformer_func(item)
        elif isinstance(item, Iterable):
            return np.array([self.transformer_func(i) for i in item])
        else:
            raise Exception('item should be integer or iterable.')

    def transform_indexes2cyclic_indexes(self, i) -> int:
        return i % self.array_len

    def transform_indexes2extend_indexes(self, i) -> int:
        return max((0, min((i, self.array_len - 1))))

    def get_transformed_closedrange(self, indexi, indexf):
        return np.array(self[np.arange(indexi, indexf + 1)])


class ArrayIndexerNd:
    def __init__(self, array: Union[Tuple, List, np.ndarray], modes: (str, Tuple[str]) = CYCLIC):
        self.array_shape = array if isinstance(array, tuple) else np.shape(array)
        self.modes = (modes,) * len(self.array_shape) if isinstance(modes, str) else modes
        assert len(self.array_shape) == len(self.modes)
        assert all([mode == EXTEND or mode == CYCLIC or REFLECT for mode in self.modes]), 'mode should be extend or cyclic'
        self.transformer_func_list = [getattr(self, 'transform_indexes2{}_indexes'.format(mode)) for mode in self.modes]

    def __len__(self):
        return self.array_shape

    def transform_single_dimension(self, indexes: Union[int, List, np.ndarray], dimension):
        if isinstance(indexes, int):
            return self.transformer_func_list[dimension](indexes, dimension)
        elif isinstance(indexes, (List, np.ndarray)):
            return [self.transformer_func_list[dimension](ix, dimension) for ix in indexes]
        else:
            raise Exception("Type not implemented.")

    def __getitem__(self, item):
        if isinstance(item, Iterable):
            if isinstance(item[0], Integral):
                res = tuple([int(transformer_func(i, dim)) for dim, (i, transformer_func) in
                             enumerate(zip(item, self.transformer_func_list))])
            elif isinstance(item, Iterable):
                res = tuple([transformer_func(i, dim) for dim, (i, transformer_func) in
                             enumerate(zip(np.array(item).T, self.transformer_func_list))])
            else:
                raise Exception('item should be integer or iterable.')
        else:
            raise Exception('item should be iterable.')
        return res

    def transform_indexes2cyclic_indexes(self, i, dim) -> int:
        return i % self.array_shape[dim]

    def transform_indexes2extend_indexes(self, i, dim) -> int:
        i = i if isinstance(i, Iterable) else [i]
        return np.array(np.max(
            (
                np.zeros((len(i), 1)),
                np.min(
                    (
                        np.reshape(i, (-1, 1)),
                        np.array([self.array_shape[dim] - 1] * len(i)).reshape((-1, 1))
                    ),
                    axis=0)
            ),
            axis=0
        ), dtype=int).ravel()

    def transform_indexes2reflect_indexes(self, i, dim) -> int:
        return self.array_shape[dim] - i % self.array_shape[dim] if i >= self.array_shape[dim] else np.abs(i)

    def get_transformed_closedrange(self, indexi: Tuple, indexf: Tuple):
        return np.array(self[list(itertools.product(*[np.arange(ii, ie + 1) for ii, ie in zip(indexi, indexf)]))])
