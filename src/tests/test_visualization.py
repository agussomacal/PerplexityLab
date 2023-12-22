import os
import sys
import time
import unittest

import numpy as np
import seaborn as sns

from PerplexityLab.DataManager import DataManager, JOBLIB
from PerplexityLab.LabPipeline import LabPipeline, FunctionBlock
from PerplexityLab.visualization import generic_plot, make_data_frames, perplex_plot, one_line_iterator, \
    get_path_name2replot_data


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

    def test_group_by_plot(self):
        @perplex_plot(group_by="k")
        def plot(fig, ax, x, z, k):
            ax.scatter(x, z, label=f"k={k}")

        paths = plot(self.data_manager, z=lambda x, y: y / x,
                     axis_font_dict={'color': 'black', 'weight': 'normal', 'size': 20},
                     legend_font_dict={'weight': 'normal', "size": 20, 'stretch': 'normal'}, )
        assert len(paths) == 1
        assert all([isinstance(path, str) for path in paths])

    def test_group_by_plot_sort_by(self):
        @perplex_plot(group_by="k")
        def plot(fig, ax, x, z, k):
            ax.plot(x, z, label=f"k={k}")

        paths = plot(self.data_manager, name="sort_test_2", z=lambda x, y: y / x, sort_by=["z"],
                     xlabel="X", ylabel="Y", xticks=[-0.5, 0, 0.7],
                     axis_font_dict={'color': 'black', 'weight': 'bold', 'size': 25},
                     )
        assert len(paths) == 1
        assert all([isinstance(path, str) for path in paths])

    def test_create_preimage_data(self):
        @perplex_plot(group_by="k")
        def plot(fig, ax, x, z, k):
            ax.plot(x, z, label=f"k={k}")

        if os.path.exists(get_path_name2replot_data(self.data_manager, plot, "sort_test_2", JOBLIB)):
            os.remove(get_path_name2replot_data(self.data_manager, plot, "sort_test_2", JOBLIB))

        t0 = time.time()
        paths = plot(self.data_manager, name="sort_test_2", z=lambda x, y: y / x, sort_by=["z"],
                     xlabel="X", ylabel="Y", xticks=[-0.5, 0, 0.7],
                     axis_font_dict={'color': 'black', 'weight': 'bold', 'size': 25},
                     create_preimage_data=True
                     )
        t1 = time.time() - t0
        t0 = time.time()
        paths = plot(self.data_manager, name="sort_test_2", z=lambda x, y: y / x, sort_by=["z"],
                     xlabel="X", ylabel="Y", xticks=[-0.5, 0, 0.7],
                     axis_font_dict={'color': 'black', 'weight': 'bold', 'size': 25},
                     create_preimage_data=True
                     )
        t2 = time.time() - t0
        print(t2 - t1)
        assert len(paths) == 1
        assert all([isinstance(path, str) for path in paths])
        assert t2 < t1
        paths = plot(self.data_manager, name="nonexistentplot", z=lambda x, y: y / x, sort_by=["z"],
                     xlabel="X", ylabel="Y", xticks=[-0.5, 0, 0.7],
                     axis_font_dict={'color': 'black', 'weight': 'bold', 'size': 25},
                     create_preimage_data=True,
                     only_create_preimage_data=True
                     )
        assert len(paths) == 0
        assert os.path.exists(get_path_name2replot_data(self.data_manager, plot, "nonexistentplot", JOBLIB))
        assert not os.path.exists(
            get_path_name2replot_data(self.data_manager, plot, "nonexistentplot", JOBLIB).replace("data2replot_", ""))

    def test_one_line_iterator(self):
        @perplex_plot()
        @one_line_iterator
        def one_plot(fig, ax, x, z):
            ax.scatter(x, z)

        paths = one_plot(self.data_manager, axes_by=["k"], z=lambda x, y: y / x)
        assert len(paths) == 1
        assert all([isinstance(path, str) for path in paths])

        @perplex_plot()
        @one_line_iterator
        def one_plot(fig, ax, x, z, a=1):
            ax.scatter(x, z + a)

        paths = one_plot(self.data_manager, axes_by=["k"], z=lambda x, y: y / x)
        assert len(paths) == 1
        assert all([isinstance(path, str) for path in paths])

    def test_plot_versus(self):
        paths = generic_plot(self.data_manager, x="x", y="z", label="preprocessing", plot_func=sns.lineplot,
                             z=lambda x, y: y / x, axes_by=["k"])
        assert len(paths) == 1
        assert all([isinstance(path, str) for path in paths])

    def test_plot_versus_sort(self):
        paths = generic_plot(self.data_manager, name="sort_test", x="x", y="z", label="preprocessing",
                             plot_func=sns.lineplot,
                             z=lambda x, y: y / x, axes_by=["k"], sort_by=["z"]
                             )
        assert len(paths) == 1
        assert all([isinstance(path, str) for path in paths])

    def test_plot_versus_label_outside(self):
        paths = generic_plot(self.data_manager, name="label_outside", x="x", y="z", label="preprocessing",
                             plot_func=sns.lineplot,
                             z=lambda x, y: y / x, axes_by=["k"], sort_by=["z"],
                             axis_font_dict={'color': 'black', 'weight': 'normal', 'size': 30},
                             legend_font_dict={'weight': 'normal', "size": 30, 'stretch': 'normal'},
                             legend_outside=True, legend_loc="lower center"
                             )
        assert len(paths) == 1
        assert all([isinstance(path, str) for path in paths])

    def test_plot_versus_folder(self):
        paths = generic_plot(self.data_manager, x="x", y="z", label="preprocessing", plot_func=sns.lineplot,
                             z=lambda x, y: y / x, folder_by=["k"])
        assert len(paths) == 2
        assert "/k0/" in paths[0] + paths[1] and "/k1/" in paths[0] + paths[1]
        assert all([isinstance(path, str) for path in paths])

    def test_plot_versus_folderpath(self):
        paths = generic_plot(self.data_manager, path=[self.data_manager.path.joinpath("pepe")], x="x", y="z",
                             label="preprocessing", plot_func=sns.lineplot,
                             z=lambda x, y: y / x, folder_by=["k"])
        assert len(paths) == 4
        assert "/k0/" in paths[0] and "/k0/" in paths[1]
        assert "/k1/" in paths[2] and "/k1/" in paths[3]
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
