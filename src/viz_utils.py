import inspect
import os
from contextlib import contextmanager
from pathlib import Path

import matplotlib.pylab as plt
import numpy as np
from tqdm import tqdm

from src.DataManager import DataManager, group, dmfilter
from src.performance_utils import timeit, get_map_function

INCHES_PER_LETTER = 0.11
INCHES_PER_LABEL = 0.3
LEGEND_EXTRA_PERCENTAGE_SPACE = 0.1


def perplex_plot(plot_function):
    def decorated_func(data_manager: DataManager, name="", folder="", plot_by=[], axes_by=[],
                       axes_xy_proportions=(10, 8),
                       dpi=None, plot_again=True, format=".png", num_cores=1, **kwargs):
        path = data_manager.path.joinpath(folder)
        Path(path).mkdir(parents=True, exist_ok=True)

        function_arg_names = inspect.getfullargspec(plot_function).args
        assert len({"fig", "ax"}.intersection(
            function_arg_names)) == 2, "fig and ax should be two varaibles of ploting " \
                                       "function but they were not found: {}".format(function_arg_names)
        vars4plot = set(function_arg_names).intersection(data_manager.columns)
        specified_vars = {k: v for k, v in kwargs.items() if k in data_manager.columns}
        extra_arguments = {k: v for k, v in kwargs.items() if
                           k in function_arg_names and k not in specified_vars.keys()}

        # vars4plot = set(function_arg_names).difference({"fig", "ax"}).difference(extra_arguments.keys())

        def iterator():
            for grouping_vars, data2plot in group(data_manager, names=vars4plot.union(plot_by, axes_by), by=plot_by,
                                                  **specified_vars):
                plot_name = name + plot_function.__name__ + "_" + "_".join(
                    ["{}{}".format(k, v) for k, v in grouping_vars.items()])
                plot_name = f"{path}/{plot_name}{format}"
                if plot_again or not os.path.exists(plot_name):
                    yield list(group(data2plot, names=vars4plot.union(plot_by, axes_by), by=axes_by)), plot_name

        def parallel_func(args):
            data2plot_per_plot, plot_name = args
            with timeit("Plot {}".format(plot_name)):
                with many_plots_context(N_subplots=len(data2plot_per_plot), pathname=plot_name, savefig=True,
                                        return_fig=True, axes_xy_proportions=axes_xy_proportions, dpi=dpi) as fax:
                    fig, axes = fax
                    for i, (data_of_ax, data2plot_in_ax) in enumerate(data2plot_per_plot):
                        ax = get_sub_ax(axes, i)
                        ax.set_title("".join(["{}: {}\n".format(k, v) for k, v in data_of_ax.items()]))
                        plot_function(fig=fig, ax=ax,
                                      **{k: v for k, v in data2plot_in_ax.items() if k in function_arg_names},
                                      **extra_arguments)

        for _ in get_map_function(num_cores)(parallel_func, iterator()):
            pass

    return decorated_func


def squared_subplots(N_subplots, return_fig=False, axes_xy_proportions=(4, 4)):
    if N_subplots > 0:
        nrows = int(np.sqrt(N_subplots))
        ncols = int(np.ceil(N_subplots / nrows))
        # ncols = int(np.sqrt(N_subplots))
        # nrows = int(np.ceil(N_subplots / ncols))
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True,
                               figsize=(axes_xy_proportions[0] * ncols, axes_xy_proportions[1] * nrows))
        if N_subplots == 1:
            ax = np.array(ax).reshape((1, 1))
        if len(ax.shape) == 1:
            ax = ax.reshape((1, -1))
        if return_fig:
            return fig, ax
        else:
            return ax


@contextmanager
def many_plots_context(N_subplots, pathname, savefig=True, return_fig=False, axes_xy_proportions=(4, 4),
                       dpi=None):
    figax = squared_subplots(N_subplots, return_fig=return_fig, axes_xy_proportions=axes_xy_proportions)

    yield figax

    # if filename[-4:] not in ['.png', '.jpg', '.svg']:
    #     filename += '.png'

    if savefig:
        plt.savefig(pathname, dpi=dpi)
    else:
        plt.show()
    plt.close()


@contextmanager
def save_fig(path, filename, axes_xy_proportions=(10, 8)):
    with many_plots_context(1, path, filename, savefig=True, return_fig=True, axes_xy_proportions=axes_xy_proportions,
                            dpi=None) as fax:
        yield fax


def make_gif(directory, image_list_names, gif_name, delay=20):
    ftext = '{}/image_list.txt'.format(directory)
    fp_out = "{}/{}".format(os.path.dirname(directory), gif_name)
    if fp_out[-4:] not in ['.gif']:
        fp_out = '{}.gif'.format(fp_out)

    with open(ftext, 'w') as file:
        for item in image_list_names:
            file.write("%s\n" % "{}/{}".format(directory, item))

    os.system('convert -delay {} @{} {}'.format(delay, ftext, fp_out))  # On windows convert is 'magick'
    return fp_out


# def make_gif(directory, image_names_list, gif_name, duration=200):
#     # filepaths
#     fp_in = "{}/*.png".format(directory)
#     fp_out = "{}/{}.gif".format(os.path.dirname(directory), gif_name)
#
#     # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
#     img, *imgs = [Image.open(os.path.join(directory, f)) for f in image_names_list]
#
#     print('Doing gif')
#     img.save(fp=fp_out, format='GIF', append_images=imgs,
#              save_all=True, duration=duration, loop=0)
#     os.remove(directory)


def get_sub_ax(ax, i):
    nrows, ncols = ax.shape
    return ax[i // ncols, i % ncols]
