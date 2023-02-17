import os
import unittest

from benedict import benedict

from src.DataManager import DataManager, JOBLIB, ALL


class TestDataManager(unittest.TestCase):
    def setUp(self) -> None:
        self.dm = DataManager(
            path=os.path.abspath(os.path.join(__file__, os.pardir)),
            name="TestDataManager",
            format=JOBLIB
        )

    def test_basic(self):
        self.dm.add_result(input_params={"a": 1, "b": 2}, input_funcs=dict(), function_block="block1",
                           function_name="f_name", function_result={"res": 5})
        # Recognized distinct parameters
        assert len(self.dm.parameters) == 2
        assert set(self.dm.parameters.keys()) == {"a", "b"}
        assert self.dm.parameters["a"].values == {1}
        assert self.dm.parameters["b"].values == {2}

    def test_getitem(self):
        self.dm.add_result(input_params={"a": 1, "b": 2}, input_funcs=dict(), function_block="block1",
                           function_name="f_name", function_result={"res": 5})

        # test for param
        assert self.dm["a"] == [1]
        # test for block
        assert self.dm["block1"] == ["f_name"]
        assert self.dm[["block1"]] == {"block1": ["f_name"]}
        # test for var
        assert self.dm[["res"]] == {"res": [5]}
        assert self.dm["res"] == [5]
        # test for mixed
        assert self.dm[["res", "a"]] == {"res": [5], "a": [1]}
        assert self.dm[ALL] == {"a": [1], "b": [2], "block1": ["f_name"], "res": [5]}

    def test_multiple_getitem(self):
        self.dm.add_result(input_params={"a": 1, "b": 2}, input_funcs=dict(), function_block="block1",
                           function_name="f_name", function_result={"res": 5})
        self.dm.add_result(input_params={"a": 2, "b": 2}, input_funcs=dict(), function_block="block1",
                           function_name="f_name", function_result={"res": 2})
        self.dm.add_result(input_params={"a": 2, "b": 5}, input_funcs=dict(), function_block="block1",
                           function_name="f_name", function_result={"res": 1})

        # test for param
        assert self.dm["a"] == [1, 2, 2]
        assert self.dm["b"] == [2, 2, 5]
        assert self.dm[["a", "b"]] == {"a": [1, 2, 2], "b": [2, 2, 5]}
        # # test for block
        assert self.dm["block1"] == ["f_name", "f_name", "f_name"]
        assert self.dm[["block1"]] == {"block1": ["f_name", "f_name", "f_name"]}
        # # test for var
        assert self.dm[["res"]] == {"res": [5, 2, 1]}
        assert self.dm["res"] == [5, 2, 1]
        # # test for mixed
        assert self.dm[["res", "a"]] == {"res": [5, 2, 1], "a": [1, 2, 2]}
        assert self.dm[ALL] == {"a": [1, 2, 2], "b": [2, 2, 5], "block1": ["f_name", "f_name", "f_name"],
                                "res": [5, 2, 1]}

        self.dm.add_result(input_params={"a": 2, "b": 5}, input_funcs={"block1": "f_name"}, function_block="block2",
                           function_name="f_name_2", function_result={"res2": 1})
        assert self.dm[ALL] == {"a": [1, 2, 2], "b": [2, 2, 5], "block1": ["f_name", "f_name", "f_name"],
                                "res": [5, 2, 1], "block2": ["f_name_2", "f_name_2", "f_name_2"],
                                "res2": [None, None, 1]}

        self.dm.add_result(input_params={"a": 2, "b": 5}, input_funcs={"block1": "f_name"}, function_block="block2",
                           function_name="f_name_22", function_result={"res2": 2})
        assert set(self.dm["res2"]) == {None, None, None, None, 1, 2}

    def test_save_joblib(self):
        self.dm.add_result(input_params={"a": 1, "b": 2}, input_funcs=dict(), function_block="block1",
                           function_name="f_name", function_result={"res": 5})
        self.dm.add_result(input_params={"a": 2, "b": 2}, input_funcs=dict(), function_block="block1",
                           function_name="f_name", function_result={"res": 2})
        self.dm.add_result(input_params={"a": 2, "b": 5}, input_funcs=dict(), function_block="block1",
                           function_name="f_name", function_result={"res": 1})
        self.dm.save()

    def test_load_joblib(self):
        self.dm.load()
        assert isinstance(self.dm.database, benedict)
        assert set(self.dm.parameters.keys()) == {"a", "b"}
        assert set(self.dm.function_blocks.keys()) == {"block1"}
        assert set(self.dm.variables.keys()) == {"res"}
        assert self.dm["res"] == [5, 2, 1]

    if __name__ == '__main__':
        unittest.main()
