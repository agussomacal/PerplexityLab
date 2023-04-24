import os
import unittest

import numpy as np
import seaborn as sns

from PerplexityLab.DataManager import DataManager, JOBLIB
from PerplexityLab.LabPipeline import LabPipeline, FunctionBlock
from PerplexityLab.visualization import generic_plot, make_data_frames


class TestVizUtils(unittest.TestCase):
    def setUp(self) -> None:
        self.data_manager = DataManager(
            path=os.path.abspath(os.path.join(__file__, os.pardir)),
            name="TestDataManager",
            format=JOBLIB
        )
        lab = LabPipeline()
        lab.define_new_block_of_functions("preprocessing",
                                          FunctionBlock(name="squared", function=lambda x, k: {"y": x ** 2 + k}),
                                          FunctionBlock(name="line", function=lambda x, k: {"y": x + 1 + k}))

        self.data_manager = lab.execute(self.data_manager,
                                        num_cores=1, forget=True, save_on_iteration=False,
                                        x=np.linspace(-1, 1), k=[0, 1])

    def test_plot_versus(self):
        paths = generic_plot(self.data_manager, x="x", y="z", label="preprocessing", plot_func=sns.lineplot,
                             z=lambda x, y: y / x, axes_by=["k"])
        assert len(paths) == 1
        assert all([isinstance(path, str) for path in paths])

    def test_plot_versus_boxplot(self):
        paths = generic_plot(self.data_manager, x="preprocessing", y="z", plot_func=sns.boxplot,
                             z=lambda x, y: y / x, axes_by=["k"])
        assert len(paths) == 1
        assert all([isinstance(path, str) for path in paths])

    def test_make_df(self):
        gv, df = list(
            zip(*list(make_data_frames(self.data_manager, var_names=["x", "z"], group_by=["k"], z=lambda x, y: y / x,
                                       preprocessing="squared"))))
        assert len(gv) == len(self.data_manager.parameters["k"].values)
        assert df[0].shape == (len(self.data_manager.parameters["x"].values), 3)

    if __name__ == '__main__':
        unittest.main()
