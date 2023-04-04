import inspect
import logging
import os
import pickle
import time
from collections import OrderedDict
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import Callable

import joblib
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


def if_true_str(var, var_name, prefix="", end=""):
    return (prefix + var_name + end) if var else ""


class NamedPartial:
    def __init__(self, func, *args, **kwargs):
        self.f = partial(func, *args, **kwargs)
        self.__name__ = func.__name__ + "_" + "_".join(list(map(str, args))) + "_".join(
            ["{}{}".format(k, v) for k, v in kwargs.items()])

    def __call__(self, *args, **kwargs):
        return self.f(*args, **kwargs)

    def __str__(self):
        return self.__name__


def if_exist_load_else_do(do_func):
    def decorated_func(path, filename=None, file_format="joblib", recalculate=False, *args, **kwargs):
        Path(path).mkdir(parents=True)
        filename = do_func.__name__ if filename is None else filename
        filepath = f"{path}/{filename}.{file_format}"
        if recalculate or not os.path.exists(filepath):
            with timeit(f"Processing {filename}:"):
                # Projet points into graph edges
                data = do_func(*args, **kwargs)

                if "npy" in file_format:
                    np.save(filepath, data)
                elif "pickle" in file_format:
                    with open(filepath, "r") as f:
                        pickle.dump(data, f)
                elif "joblib" in file_format:
                    data = joblib.dump(data, filename)
                else:
                    raise Exception(f"Format {file_format} not implemented.")
        else:
            with timeit(f"Loading pre-processed {filename}:"):
                if "npy" in file_format:
                    data = np.load(filepath)
                elif "pickle" in file_format:
                    with open(filepath, "r") as f:
                        data = pickle.load(f)
                elif "joblib" in file_format:
                    data = joblib.load(filename)
                else:
                    raise Exception(f"Format {file_format} not implemented.")
        return data

    return decorated_func
