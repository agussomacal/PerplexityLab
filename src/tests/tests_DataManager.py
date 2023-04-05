import os
import unittest

import numpy as np
from benedict import benedict

from src.DataManager import DataManager, JOBLIB, ALL, group, apply


class TestDataManager(unittest.TestCase):
    def setUp(self) -> None:
        self.dm = DataManager(
            path=os.path.abspath(os.path.join(__file__, os.pardir)),
            name="TestDataManager",
            format=JOBLIB
        )

    def add_some_results(self):
        self.dm.add_result(input_params={"a": 1, "b": 2}, input_funcs=dict(), function_block="block1",
                           function_name="f_name", function_result={"res": 5})
        self.dm.add_result(input_params={"a": 2, "b": 2}, input_funcs=dict(), function_block="block1",
                           function_name="f_name", function_result={"res": 2})
        self.dm.add_result(input_params={"a": 2, "b": 5}, input_funcs=dict(), function_block="block1",
                           function_name="f_name", function_result={"res": 1})

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
        self.add_some_results()

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

    def test_save_load_joblib(self):
        self.add_some_results()
        self.dm.save()

        self.dm.load()
        assert isinstance(self.dm.database, benedict)
        assert set(self.dm.parameters.keys()) == {"a", "b"}
        assert set(self.dm.function_blocks.keys()) == {"block1"}
        assert set(self.dm.variables.keys()) == {"res"}
        assert self.dm["res"] == [5, 2, 1]

    def test_group(self):
        self.add_some_results()
        self.dm.add_result(input_params={"a": 2, "b": 5}, input_funcs=dict(), function_block="block1",
                           function_name="f_name", function_result={"res": 1})
        for dby, d in group(self.dm, names=["a", "b"], by=["a"]):
            assert len(set(d["a"])) == 1
        for dby, d in group(self.dm, names=["a", "b"], by=[]):
            assert d["a"] == self.dm["a"]
        for dby, d in group(self.dm["a", "b"], names=["a", "b"], by=[]):
            assert d["a"] == self.dm["a"]

    def test_apply(self):
        self.add_some_results()
        self.dm.add_result(input_params={"a": 2, "b": 5}, input_funcs=dict(), function_block="block1",
                           function_name="f_name", function_result={"res": 1})
        d = apply(self.dm, names=["a"], sqa=lambda a: a ** 2)
        assert np.all(np.array(d["a"]) ** 2 == np.array(d["sqa"]))
        # test if input is dic of lists
        d = apply(self.dm[ALL], names=["a"], sqa=lambda a: a ** 2)
        assert np.all(np.array(d["a"]) ** 2 == np.array(d["sqa"]))

    def test_emissions(self):
        dfsum = self.dm.get_emissions_summary(group_by_experiment=False, group_by_layer=False)
        assert dfsum.shape == (3,)
        dfsum = self.dm.get_emissions_summary(group_by_experiment=True, group_by_layer=False)
        assert dfsum.shape == (1, 3)
        dfsum = self.dm.get_emissions_summary(group_by_experiment=True, group_by_layer=True)
        assert dfsum.shape[1] == 3
        assert self.dm.CO2kg > 1e-10
        assert self.dm.electricity_consumption_kWh > 1e-10
        assert self.dm.computation_time_s > 1e-10

    if __name__ == '__main__':
        unittest.main()
