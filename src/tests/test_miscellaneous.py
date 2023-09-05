import pickle
import unittest

from PerplexityLab.miscellaneous import ClassPartialInit


class Foo1:
    def __init__(self, a):
        self.a = a


class Foo2(Foo1):
    def __init__(self, a, b):
        super().__init__(a=a)
        self.b = b


class TestVizUtils(unittest.TestCase):
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

    if __name__ == '__main__':
        unittest.main()
