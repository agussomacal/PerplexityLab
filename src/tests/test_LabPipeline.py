import os
import unittest

import numpy as np

from src.DataManager import DataManager, JOBLIB
from src.LabPipeline import LabPipeline, FunctionBlock


class TestDataManager(unittest.TestCase):
    def setUp(self) -> None:
        self.data_manager = DataManager(
            path=os.path.abspath(os.path.join(__file__, os.pardir)),
            name="TestDataManager",
            format=JOBLIB
        )

    def test_init(self):
        LabPipeline()

    def test_add_layers(self):
        lab = LabPipeline()
        lab.define_new_block_of_functions("preprocessing",
                                          FunctionBlock(name="squared", function=lambda x: {"y": x ** 2}))
        lab.define_new_block_of_functions("experiment",
                                          FunctionBlock(name="sumprod", function=lambda x, y: {"z": x + y, "w": x * y}),
                                          FunctionBlock(name="kprod", function=lambda k, y: {"a": k * y})
                                          )

    def test_execute(self):
        lab = LabPipeline()
        lab.define_new_block_of_functions("preprocessing",
                                          FunctionBlock(name="squared", function=lambda x: {"y": x ** 2}))

        k = np.array([1.0, 2.0])
        x = np.array([1.0, 2.0])
        self.data_manager = lab.execute(self.data_manager, num_cores=1, forget=True, x=x.tolist(), k=k.tolist())

        assert np.allclose(self.data_manager["y"], np.array(self.data_manager["x"]) ** 2)
        assert len(self.data_manager["y"]) == 2
        lab.define_new_block_of_functions(
            "experiment_1",
            FunctionBlock(name="kplus", function=lambda k: {"b": k}),
        )
        lab.define_new_block_of_functions(
            "experiment_2",
            # FunctionBlock(name="kplus", function=lambda k: {"a": k + 1}),
            # FunctionBlock(name="ysubk", function=lambda k, y: {"a": y - k}),
            FunctionBlock(name="sumprod", function=lambda x, y: {"a": y / x}),
        )
        lab.define_new_block_of_functions(
            "experiment_3",
            FunctionBlock(name="ysubk", function=lambda k, y: {"c": y - k}),
        )
        self.data_manager = lab.execute(self.data_manager, num_cores=1, forget=True, x=x.tolist(), k=k.tolist())

        assert set(self.data_manager["a"] * 2) == set(self.data_manager["x"])
        assert len(self.data_manager["c"]) == len(k) * len(x)

    def test_execute_parallel(self):
        lab = LabPipeline()
        lab.define_new_block_of_functions("preprocessing",
                                          FunctionBlock(name="squared", function=lambda x, k: {"y": x ** 2 + k}))

        k = np.array([1.0, 2.0, 3.0])
        x = np.array([1.0, 2.0, 5.0])
        self.data_manager = lab.execute(self.data_manager, num_cores=-1, forget=True, x=x.tolist(), k=k.tolist())

        assert np.allclose(self.data_manager["y"],
                           np.array(self.data_manager["x"]) ** 2 + np.array(self.data_manager["k"]))
        assert len(self.data_manager["y"]) == 9

    # def test_not_save(self):
    #     lab = LabPipeline()
    #     lab.define_new_block_of_functions("preprocessing",
    #                                       FunctionBlock(name="squared", function=lambda x, k: {"y": x ** 2 + k}),
    #                                       save=False)
    #
    #     k = np.array([1.0, 2.0, 3.0])
    #     x = np.array([1.0, 2.0, 5.0])
    #     self.data_manager = lab.execute(self.data_manager, num_cores=-1, forget=True, x=x.tolist(), k=k.tolist())
    #
    #     assert np.allclose(self.data_manager["y"],
    #                        np.array(self.data_manager["x"]) ** 2 + np.array(self.data_manager["k"]))
    #     assert len(self.data_manager["y"]) == 9
    #     self.data_manager.save()
    #     self.data_manager.reset()
    #     self.data_manager.load()
    #
    #     assert len(self.data_manager.database) == 0
    #
    #     lab.define_new_block_of_functions("experiment",
    #                                       FunctionBlock(name="attack", function=lambda y, k: {"u": y ** 2 + k}),
    #                                       save=True)
    #     self.data_manager = lab.execute(self.data_manager, num_cores=-1, forget=True, x=x.tolist(), k=k.tolist())
    #     self.data_manager.save()
    #     self.data_manager.reset()
    #     self.data_manager.load()
    #     assert len(self.data_manager.database) > 0
