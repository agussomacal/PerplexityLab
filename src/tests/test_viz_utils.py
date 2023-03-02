import os
import unittest

import numpy as np
import seaborn as sns

from src.DataManager import DataManager, JOBLIB
from src.LabPipeline import LabPipeline, FunctionBlock
from src.viz_utils import squared_subplots, generic_plot


class TestUtilsSquaredSubplots(unittest.TestCase):
    def setUp(self) -> None:
        self.data_manager = DataManager(
            path=os.path.abspath(os.path.join(__file__, os.pardir)),
            name="TestDataManager",
            format=JOBLIB
        )

    def test_squared_subplots(self):
        # assert squared_subplots(1).shape == (1, 1)
        # assert squared_subplots(2).shape == (1, 2)
        # assert squared_subplots(3).shape == (1, 3)
        # assert squared_subplots(4).shape == (2, 2)
        # assert squared_subplots(5).shape == (2, 3)
        # assert squared_subplots(6).shape == (2, 3)
        pass

    def test_plot_versus(self):
        lab = LabPipeline()
        lab.define_new_block_of_functions("preprocessing",
                                          FunctionBlock(name="squared", function=lambda x, k: {"y": x ** 2 + k}),
                                          FunctionBlock(name="line", function=lambda x, k: {"y": x + 1 + k}))

        self.data_manager = lab.execute(self.data_manager,
                                        num_cores=1, forget=True, save_on_iteration=False,
                                        x=np.linspace(-1, 1), k=[0, 1])

        generic_plot(x="x", y="y", label="preprocessing", seaborn_func=sns.lineplot)(self.data_manager, axes_by=["k"])

    if __name__ == '__main__':
        unittest.main()
