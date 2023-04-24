import os
import unittest

import numpy as np
import seaborn as sns

from src.DataManager import DataManager, JOBLIB
from src.LaTexReports import Code2LatexConnector
from src.LabPipeline import LabPipeline, FunctionBlock
from src.visualization import generic_plot


class TestLaTexReports(unittest.TestCase):
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

    def test_compile_template(self):
        report = Code2LatexConnector(path=self.data_manager.path, filename='Report')
        report.create_template()

        generic_plot(self.data_manager, path=report.get_plot_path(),
                     x="x", y="z", label="preprocessing", plot_func=sns.lineplot, z=lambda x, y: y / x,
                     axes_by=["k"])

        report.compile()

    if __name__ == '__main__':
        unittest.main()
