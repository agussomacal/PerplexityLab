import inspect
import os
import pickle
import shutil
import time
from collections import OrderedDict
from contextlib import contextmanager
from functools import partial, partialmethod
from pathlib import Path
from typing import Callable, Dict

import joblib
import numpy as np
import pandas as pd
from pathos.multiprocessing import Pool, cpu_count
from scipy.sparse import csr_array, bsr_array, coo_array, csc_array, dia_array, dok_array, lil_array, bsr_matrix, \
    coo_matrix, csc_matrix, csr_matrix, dia_matrix, dok_matrix, lil_matrix


# ----------- functions -----------
def get_default_args(func):
    """
    Get default arguments of function.
    """
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


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
class DefaultOrderedDict(OrderedDict):
    # Source: http://stackoverflow.com/a/6190500/562769
    # Source: https://stackoverflow.com/questions/6190331/how-to-implement-an-ordered-default-dict
    def __init__(self, default_factory=None, *a, **kw):
        if (default_factory is not None and
                not isinstance(default_factory, Callable)):
            raise TypeError('first argument must be callable')
        OrderedDict.__init__(self, *a, **kw)
        self.default_factory = default_factory

    def __getitem__(self, key):
        try:
            return OrderedDict.__getitem__(self, key)
        except KeyError:
            return self.__missing__(key)

    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        self[key] = value = self.default_factory()
        return value

    def __reduce__(self):
        if self.default_factory is None:
            args = tuple()
        else:
            args = self.default_factory,
        return type(self), args, None, None, self.items()

    def copy(self):
        return self.__copy__()

    def __copy__(self):
        return type(self)(self.default_factory, self)

    def __deepcopy__(self, memo):
        import copy
        return type(self)(self.default_factory,
                          copy.deepcopy(self.items()))

    def __repr__(self):
        return 'OrderedDefaultDict(%s, %s)' % (self.default_factory,
                                               OrderedDict.__repr__(self))


def filter_dict(keys, kwargs: Dict):
    """
    filter_dict receives a dict (so it can accept non string keys)
    """
    return OrderedDict([(k, kwargs[k]) for k in keys if k in kwargs])


def filter_dict4func(function, **kwargs):
    ins = inspect.getfullargspec(function)
    return filter_dict(ins.args + ins.kwonlyargs, kwargs)
    # return filter_dict(inspect.getfullargspec(function).args, kwargs)


def partial_filter(function, **kwargs):
    return partial(function, **filter_dict4func(function, **kwargs))


class NamedPartial:
    def __init__(self, func, *args, **kwargs):
        self.f = partial(func, *args, **kwargs)
        self.__name__ = func.__name__  # + "_" + "_".join(list(map(str, args))) + "_".join(
        # ["{}{}".format(k, v) for k, v in kwargs.items()])

    def add_prefix_to_name(self, prefix):
        self.__name__ = str(prefix) + self.__name__
        return self

    def add_sufix_to_name(self, sufix):
        self.__name__ = self.__name__ + str(sufix)
        return self

    def __call__(self, *args, **kwargs):
        return self.f(*args, **kwargs)

    def __str__(self):
        return self.__name__


def ClassPartialInit(class_type, class_name=None, **kwargs):
    if class_name is None:
        class_name = "Partial" + class_type.__name__ + "_".join([str(k) + str(v) for k, v in kwargs.items()])
    new_class = type(
        class_name,
        (class_type,) + class_type.__bases__,
        {
            '__init__': partialmethod(class_type.__init__, **kwargs),
            # '__reduce__': lambda self: (class_type, tuple(), self.__dict__.copy())
        })
    globals()[class_name] = new_class
    return new_class


# ---------- File utils ---------- #
def copy_main_script_version(file, results_path):
    shutil.copyfile(os.path.realpath(file), f"{results_path}/main_script.py")


def check_create_path(path, *args):
    path = Path(path)
    for name in args:
        path = path.joinpath(name)
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def clean_str4saving(s):
    return s.replace("(", "").replace(")", "").replace("[", "").replace("]", "").replace(",", "").replace(
        ".", "").replace(";", "").replace(":", "").replace(" ", "_")


import copy

DictProxyType = type(object.__dict__)


def make_hash(o):
    """
    Thanks to: https://stackoverflow.com/questions/5884066/hashing-a-dictionary

    Makes a hash from a dictionary, list, tuple or set to any level, that
    contains only other hashable types (including any lists, tuples, sets, and
    dictionaries). In the case where other kinds of objects (like classes) need
    to be hashed, pass in a collection of object attributes that are pertinent.
    For example, a class can be hashed in this fashion:

      make_hash([cls.__dict__, cls.__name__])

    A function can be hashed like so:

      make_hash([fn.__dict__, fn.__code__])
    """

    if type(o) == DictProxyType:
        o2 = {}
        for k, v in o.items():
            if not k.startswith("__"):
                o2[k] = v
        o = o2

    if isinstance(o, (set, tuple, list)):
        return hash(tuple([make_hash(e) for e in o]))
    elif isinstance(o, pd.DataFrame):
        return make_hash((o.values, o.columns.tolist(), o.index.tolist()))
    elif isinstance(o, np.ndarray):
        return make_hash(o.tolist())
    elif isinstance(o, (
            csr_array, bsr_array, coo_array, csc_array, dia_array, dok_array, lil_array, bsr_matrix, coo_matrix,
            csc_matrix,
            csr_matrix, dia_matrix, dok_matrix, lil_matrix)):
        return make_hash((o.nonzero(), o.data))
    elif not isinstance(o, dict):
        return hash(o)

    new_o = copy.deepcopy(o)
    for k, v in new_o.items():
        new_o[k] = make_hash(v)

    return hash(tuple(frozenset(sorted(new_o.items()))))


def ifex_saver(data, filepath, saver, file_format):
    with timeit(f"Saving processed {filepath}:"):
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


def ifex_loader(filepath, loader, file_format):
    """
    :param file_format: format for the termination of the file. If not known specify loader an saver. known ones are: npy, pickle, joblib
    :param loader: function that knows how to load the file
    """
    with timeit(f"Loading pre-processed {filepath}:"):
        if loader is not None:
            data = loader(filepath)
        elif "npy" == file_format:
            data = np.load(filepath, allow_pickle=True)
        elif "pickle" == file_format:
            with open(filepath, "r") as f:
                data = pickle.load(f)
        elif "joblib" == file_format:
            data = joblib.load(filepath)
        else:
            raise Exception(f"Format {file_format} not implemented.")
        return data


def if_exist_load_else_do(file_format="joblib", loader=None, saver=None, description=None, check_hash=False):
    """
    Decorator to manage loading and saving of files after a first processing execution.
    :param file_format: format for the termination of the file. If not known specify loader an saver. known ones are: npy, pickle, joblib
    :param loader: function that knows how to load the file
    :param saver: function that knows how to save the file
    :param description: description of data as a function depending on the type of data.
    :return:
    a function with the same aprameters as the decorated plus
    :path to specify path to folder
    :filename=None to specify filename
    :recalculate=False to force recomputation.
    :check_hash=False to recalculate if inputs to function change with respect of saved data
    """

    def decorator(do_func):
        def decorated_func(path, filename=None, recalculate=False, *args, **kwargs):
            path = Path(path)
            path.mkdir(parents=True, exist_ok=True)
            filename = do_func.__name__ if filename is None else filename
            # if args4name == "all":
            #     args4name = sorted(kwargs.keys())
            # filename = f"{filename}" + "_".join(f"{arg_name}{kwargs[arg_name]}" for arg_name in args4name)
            filename = clean_str4saving(filename)
            filepath = f"{path}/{filename}.{file_format}"
            filepath_hash = f"{path}/{filename}.hash"

            # get hash of old and new file
            if check_hash:
                hash_of_input_old = None
                if os.path.exists(filepath_hash):
                    with open(filepath_hash, "r") as f:
                        hash_of_input_old = int(f.readline())
                hash_of_input = make_hash((args, kwargs))
                not_same_hash = (hash_of_input != hash_of_input_old)
            else:
                not_same_hash = True

            # process save or load
            if recalculate or not os.path.exists(filepath) or (check_hash and not_same_hash):
                # Processing
                with timeit(f"Processing {filepath}:"):
                    data = do_func(*args, **kwargs)

                # Saving data and hash
                ifex_saver(data, filepath=filepath, saver=saver, file_format=file_format)
                if check_hash:
                    with open(filepath_hash, "w") as f:
                        f.writelines(str(hash_of_input))
            else:
                # loading data
                data = ifex_loader(filepath=filepath, loader=loader, file_format=file_format)

            # do post processing
            if isinstance(description, Callable):
                description(data)
            return data

        return decorated_func

    return decorator
