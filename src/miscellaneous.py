import inspect
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


# ----------- time -----------
@contextmanager
def timeit(msg):
    print(msg, end='')
    t0 = time.time()
    yield
    print('\r -> duracion {}: {:.2f}s'.format(msg, time.time() - t0))


def calculate_time(func: Callable):
    def new_func(*args, **kwargs):
        print(f"calculating {func.__name__}")
        t0 = time.time()
        res = func(*args, **kwargs)
        t = time.time() - t0
        print(f"time spent: {t}")
        return t, res

    return new_func


# ---------- Parallel or not parallel ----------

def get_workers(workers):
    if workers > 0:
        return min((cpu_count() - 1, workers))
    else:
        return max((1, cpu_count() + workers))


def get_appropiate_number_of_workers(workers, n):
    return int(np.max((1, np.min((cpu_count() - 1, n, workers)))))


def get_map_function(workers=1):
    return map if workers == 1 else Pool(get_workers(workers)).imap_unordered


# ----------- Strings and Booleans -----------
def is_string_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def if_true_str(var, var_name, prefix="", end=""):
    return (prefix + var_name + end) if var else ""


# ---------- Dictionaries and partial evaluation of functions ---------

def filter_dict(keys, **kwargs):
    return OrderedDict([(k, v) for k, v in kwargs.items() if k in keys])


def partial_filter(function, **kwargs):
    return function(**filter_dict(inspect.getfullargspec(function).args, **kwargs))


class NamedPartial:
    def __init__(self, func, *args, **kwargs):
        self.f = partial(func, *args, **kwargs)
        self.__name__ = func.__name__ + "_" + "_".join(list(map(str, args))) + "_".join(
            ["{}{}".format(k, v) for k, v in kwargs.items()])

    def __call__(self, *args, **kwargs):
        return self.f(*args, **kwargs)

    def __str__(self):
        return self.__name__


# ---------- File utils ---------- #

def check_create_path(path, *args):
    path = Path(path)
    for name in args:
        path = path.joinpath(name)
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def clean_str4saving(s):
    return s.replace("(", "").replace(")", "").replace("[", "").replace("]", "").replace(",", "").replace(
        ".", "").replace(";", "").replace(":", "").replace(" ", "_")


def if_exist_load_else_do(file_format="joblib", loader=None, saver=None):
    """
    Decorator to manage loading and saving of files after a first processing execution.
    :param file_format: format for the termination of the file. If not known specify loader an saver
    :param loader: function that knows how to load the file
    :param saver: function that knows how to save the file
    :return:
    """

    def decorator(do_func):
        def decorated_func(path, filename=None, recalculate=False, *args, **kwargs):
            filepath = Path(path)
            filepath.mkdir(parents=True, exist_ok=True)
            filename = do_func.__name__ if filename is None else filename
            # if args4name == "all":
            #     args4name = sorted(kwargs.keys())
            # filename = f"{filename}" + "_".join(f"{arg_name}{kwargs[arg_name]}" for arg_name in args4name)
            filename = clean_str4saving(filename)
            filepath = f"{filepath}/{filename}.{file_format}"
            if recalculate or not os.path.exists(filepath):
                with timeit(f"Processing {filepath}:"):
                    # Projet points into graph edges
                    data = do_func(*args, **kwargs)

                    if saver is not None:
                        saver(data, filepath)
                    elif "npy" in file_format:
                        np.save(filepath, data)
                    elif "pickle" in file_format:
                        with open(filepath, "r") as f:
                            pickle.dump(data, f)
                    elif "joblib" in file_format:
                        joblib.dump(data, filepath)
                    else:
                        raise Exception(f"Format {file_format} not implemented.")
            else:
                with timeit(f"Loading pre-processed {filepath}:"):

                    if loader is not None:
                        data = loader(filepath)
                    elif "npy" == file_format:
                        data = np.load(filepath)
                    elif "pickle" == file_format:
                        with open(filepath, "r") as f:
                            data = pickle.load(f)
                    elif "joblib" == file_format:
                        data = joblib.load(filepath)
                    else:
                        raise Exception(f"Format {file_format} not implemented.")
            return data

        return decorated_func

    return decorator