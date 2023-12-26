import os.path
import pickle
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from PerplexityLab.miscellaneous import ClassPartialInit, if_exist_load_else_do


class Foo1:
    def __init__(self, a):
        self.a = a


class Foo2(Foo1):
    def __init__(self, a, b):
        super().__init__(a=a)
        self.b = b


class TestVizUtils(unittest.TestCase):
    def setUp(self) -> None:
        self.path = Path(__file__).parent.joinpath("TestMicellaneous")

    def test_ClassPartial(self):
        Foo3 = ClassPartialInit(Foo2, b=3)
        f3 = Foo3(a=1)
        assert hasattr(f3, "b")
        assert f3.b == 3
        assert f3.a == 1
        assert f3.__class__.__bases__ == (Foo2, Foo1)
        assert Foo3.__bases__ == (Foo2, Foo1)

        # test pickling
        with open('ClassPartialInit.pickle', 'wb') as f:
            pickle.dump(f3, f)
        with open('ClassPartialInit.pickle', 'rb') as f:
            pickled = pickle.load(f)
        assert hasattr(pickled, "b")
        assert pickled.b == 3
        assert pickled.a == 1
        assert pickled.__class__.__bases__ == (Foo2, Foo1)
        assert Foo3.__bases__ == (Foo2, Foo1)

        Foo3 = ClassPartialInit(Foo2, class_name="Foo3", b=3)
        f3 = Foo3(a=1)
        # test pickling
        with open('ClassPartialInit.pickle', 'wb') as f:
            pickle.dump(f3, f)
        with open('ClassPartialInit.pickle', 'rb') as f:
            pickled = pickle.load(f)
        assert hasattr(pickled, "b")
        assert pickled.b == 3
        assert pickled.a == 1
        assert pickled.__class__.__bases__ == (Foo2, Foo1)
        assert Foo3.__bases__ == (Foo2, Foo1)

    def test_if_exist_load_else_do(self):
        @if_exist_load_else_do(file_format="joblib", loader=None, saver=None,
                               description=lambda data: print(data), check_hash=False)
        def do_something(a, b):
            return a + b

        path2file = f"{self.path}/do_something.joblib"
        if os.path.exists(path2file):
            os.remove(path2file)

        path2hash = f"{self.path}/do_something.hash"
        if os.path.exists(path2hash):
            os.remove(path2hash)

        res = do_something(path=self.path, filename=None, recalculate=False, a=1, b=2)
        assert os.path.exists(path2file)
        assert os.path.exists(path2hash)
        assert res == 3
        res = do_something(path=self.path, filename=None, recalculate=False, a=1, b=2)
        assert res == 3

        # change input but check_hash = False
        res = do_something(path=self.path, filename=None, recalculate=False, a=1, b=5)
        assert res == 3

    def test_if_exist_load_else_do_hash(self):
        @if_exist_load_else_do(file_format="joblib", loader=None, saver=None,
                               description=lambda data: print(data), check_hash=True)
        def do_something(a, b):
            return a + b

        path2file = f"{self.path}/do_something.joblib"
        if os.path.exists(path2file):
            os.remove(path2file)

        path2hash = f"{self.path}/do_something.hash"
        if os.path.exists(path2hash):
            os.remove(path2hash)

        res = do_something(path=self.path, filename=None, recalculate=False, a=1, b=2)
        assert os.path.exists(path2file)
        assert os.path.exists(path2hash)
        assert res == 3
        res = do_something(path=self.path, filename=None, recalculate=False, a=1, b=2)
        assert res == 3

        # change input but check_hash = False
        res = do_something(path=self.path, filename=None, recalculate=False, a=1, b=5)
        assert res == 6

    def test_if_exist_load_else_do_hash_DataFrame(self):
        @if_exist_load_else_do(file_format="joblib", loader=None, saver=None,
                               description=lambda data: print(data), check_hash=True)
        def do_something_df(a, b):
            return a + b

        # data frame
        res = do_something_df(path=self.path, filename=None,
                              a=pd.DataFrame([[2, 3]], columns=["a", "b"]),
                              b=pd.DataFrame([[2, 3]], columns=["a", "b"]))
        assert np.allclose(res.values, [[4, 6]])
        res = do_something_df(path=self.path, filename=None,
                              a=pd.DataFrame([[2, 3]], columns=["a", "b"]),
                              b=pd.DataFrame([[2, 7]], columns=["a", "b"]))
        assert np.allclose(res.values, [[4, 10]])
        res = do_something_df(path=self.path, filename=None,
                              a=pd.DataFrame([[2, 3]], columns=["a", "b"]),
                              b=pd.DataFrame([[2, 7]], columns=["a", "b"]))
        assert np.allclose(res.values, [[4, 10]])

    if __name__ == '__main__':
        unittest.main()
