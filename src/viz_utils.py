import inspect
import os
from contextlib import contextmanager

import matplotlib.pylab as plt
import numpy as np

from src.DataManager import DataManager

INCHES_PER_LETTER = 0.11  # 0.11
INCHES_PER_LABEL = 0.3
LEGEND_EXTRA_PERCENTAGE_SPACE = 0.1


def perplex_plot(plot_function):
    def decorated_func(data_manager: DataManager, folder=""):
        function_arg_names = inspect.getfullargspec(plot_function).args
        with save_fig(path=data_manager.path.joinpath(folder), filename=plot_function.__name__) as fax:
            plot_function(*fax, **data_manager[set(function_arg_names).difference({"fig", "ax"})])
    return decorated_func


@contextmanager
def save_fig(path, filename):
    fig, ax = plt.subplots()
    yield fig, ax
    plt.savefig(f"{path}/{filename}")


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
def many_plots_context(N_subplots, pathplot, savefig=True, return_fig=False, axes_xy_proportions=(4, 4), dpi=None):
    figax = squared_subplots(N_subplots, return_fig=return_fig, axes_xy_proportions=axes_xy_proportions)

    yield figax

    end = ''
    if pathplot[-4:] not in ['.png', '.jpg', '.svg']:
        end = '.png'
    if savefig:
        plt.savefig('{}{}'.format(pathplot, end), dpi=dpi)
    else:
        plt.show()
    plt.close()


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
