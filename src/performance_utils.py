import inspect
import logging
import time
from collections import OrderedDict
from contextlib import contextmanager
from typing import Callable

import numpy as np
from pathos.multiprocessing import Pool, cpu_count


@contextmanager
def timeit(msg):
    print(msg, end='')
    t0 = time.time()
    yield
    logging.info('\r -> duracion {}: {:.2f}s'.format(msg, time.time() - t0))


def calculate_time(func: Callable):
    def new_func(*args, **kwargs):
        print(f"calculating {func.__name__}")
        t0 = time.time()
        res = func(*args, **kwargs)
        t = time.time() - t0
        logging.info(f"time spent: {t}")
        return t, res

    return new_func


def get_workers(workers):
    if workers > 0:
        return min((cpu_count() - 1, workers))
    else:
        return max((1, cpu_count() + workers))


def get_map_function(workers=1):
    return map if workers == 1 else Pool(get_workers(workers)).imap_unordered


def is_string_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def get_appropiate_number_of_workers(workers, n):
    return int(np.max((1, np.min((cpu_count() - 1, n, workers)))))


def filter_dict(keys, **kwargs):
    return OrderedDict([(k, v) for k, v in kwargs.items() if k in keys])


def partial_filter(function, **kwargs):
    return function(**filter_dict(inspect.getfullargspec(function).args, **kwargs))
