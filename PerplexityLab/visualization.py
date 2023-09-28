import copy
import inspect
import itertools
import os
import warnings
from contextlib import contextmanager
from inspect import signature
from pathlib import Path
from typing import Callable, List, Union

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import seaborn as sns
from makefun import with_signature, wraps

from PerplexityLab.DataManager import DataManager, group, apply, dmfilter
from PerplexityLab.miscellaneous import timeit, get_map_function, clean_str4saving, filter_dict

INCHES_PER_LETTER = 0.11
INCHES_PER_LABEL = 0.3
LEGEND_EXTRA_PERCENTAGE_SPACE = 0.1


def plot_test(plot_function):
    def decorated_func(path: Path, name="", folder="", format=".png", axes_xy_proportions=(10, 8), dpi=None, **kwargs):
        path = path.joinpath(folder)
        Path(path).mkdir(parents=True, exist_ok=True)
        plot_name = name + plot_function.__name__
        plot_name = clean_str4saving(plot_name)
        plot_name = f"{path}/{plot_name}{format}"
        with timeit("Plot {}".format(plot_name)):
            with many_plots_context(N_subplots=1, pathname=plot_name, savefig=True,
                                    return_fig=True, axes_xy_proportions=axes_xy_proportions, dpi=dpi) as fax:
                fig, axes = fax
                plot_function(fig=fig, ax=axes[0, 0], **kwargs)
        return plot_name

    return decorated_func


def make_data_frames(data_manager: DataManager, var_names=[], group_by=[], **kwargs):
    var_names = set(var_names)
    group_by = set(group_by)
    specified_vars = {k: v if isinstance(v, list) else [v] for k, v in kwargs.items() if k in data_manager.columns}
    functions2apply = {k: v for k, v in kwargs.items() if
                       k in var_names | group_by and k not in specified_vars.keys() and isinstance(v, Callable)
                       and set(inspect.getfullargspec(v).args).issubset(data_manager.columns)}

    dm = apply(data_manager,
               names=set(var_names).union(specified_vars.keys(), group_by).difference(functions2apply.keys()),
               **functions2apply)
    var_names.update(functions2apply.keys())

    for grouping_vars, data2plot in group(dm, names=var_names.union(group_by), by=group_by, **specified_vars):
        yield grouping_vars, pd.DataFrame.from_dict(data2plot)


def one_line_iterator(plot_function):
    """
    Un group the many experiments that could come together to plot them one by one.
    """
    pf_signature = signature(plot_function)
    params2unlist = [p for p, v in pf_signature.parameters.items() if
                     v._default == inspect._empty and p not in ["fig", "ax"]]

    # TODO: when a not default variable is added this crashes because it assumes it comes from datamanager.
    @wraps(plot_function)
    def new_func(**kwargs):
        dm_params_in_plot = filter_dict(params2unlist, kwargs)
        for one_line in zip(*list(dm_params_in_plot.values())):
            kwargs.update(dict(zip(dm_params_in_plot.keys(), one_line)))
            plot_function(**kwargs)

    return new_func


def perplex_plot(plot_by_default=[], axes_by_default=[], folder_by_default=[], legend=True):
    def wraper(plot_function):
        def decorated_func(data_manager: DataManager, path=None, name=None, folder="", plot_by=plot_by_default,
                           axes_by=axes_by_default, folder_by=folder_by_default, axes_xy_proportions=(10, 8),
                           savefig=True,
                           dpi=None, plot_again=True, format=".png", num_cores=1, add_legend=legend, xlabel=None,
                           ylabel=None, **kwargs):
            format = format if format[0] == "." else "." + format

            # TODO: make it work
            if num_cores > 1:
                warnings.warn("Doesn not work for multiplecores when passing a lambda function.")
            with data_manager.track_emissions("figures"):
                # define where the plot will be done, maybe multiple places.
                default_path = data_manager.path.joinpath(folder) if folder != "" else data_manager.path
                if path is None:
                    paths = [default_path]
                elif isinstance(path, list):
                    paths = path + [default_path]
                else:
                    paths = [path, default_path]
                paths = [Path(p) for p in paths]
                [p.mkdir(parents=True, exist_ok=True) for p in paths]

                plot_by = plot_by if isinstance(plot_by, list) else [plot_by]
                axes_by = axes_by if isinstance(axes_by, list) else [axes_by]
                folder_by = folder_by if isinstance(folder_by, list) else [folder_by]

                function_arg_names = inspect.getfullargspec(plot_function).args
                assert len({"fig", "ax"}.intersection(
                    function_arg_names)) == 2, "fig and ax should be two varaibles of ploting " \
                                               "function but they were not found: {}".format(function_arg_names)
                vars4plot = set(function_arg_names).intersection(data_manager.columns)
                specified_vars = {k: v if isinstance(v, list) else [v] for k, v in kwargs.items() if
                                  k in data_manager.columns}
                functions2apply = {k: v for k, v in kwargs.items() if
                                   k in function_arg_names + plot_by + axes_by + folder_by and k not in specified_vars.keys() and isinstance(
                                       v, Callable)
                                   and set(inspect.getfullargspec(v).args).issubset(data_manager.columns)}
                functions2apply_var_needs = set(itertools.chain(*[
                    inspect.getfullargspec(v).args for k, v in kwargs.items() if
                    k in function_arg_names + plot_by + axes_by + folder_by and k not in specified_vars.keys() and isinstance(
                        v, Callable) and set(inspect.getfullargspec(v).args).issubset(data_manager.columns)]))
                extra_arguments = {k: v for k, v in kwargs.items() if
                                   k in function_arg_names and k not in specified_vars.keys()
                                   and k not in functions2apply.keys()}
                names = vars4plot.union(plot_by, axes_by, folder_by, specified_vars.keys()).difference(
                    functions2apply.keys()).union(
                    functions2apply_var_needs)
                dm = dmfilter(data_manager, names, **specified_vars)  # filter first by specified_vars
                dm = apply(dm, names=names, **functions2apply)  # now apply the functions
                vars4plot.update(functions2apply.keys())

                def iterator():
                    for grouping_vars_folder, data2plot_folder in \
                            group(dm, names=vars4plot.union(plot_by, axes_by, folder_by), by=folder_by,
                                  **specified_vars):
                        for path2plot in paths:
                            if len(folder_by) > 0:
                                folder_name = clean_str4saving(
                                    "_".join(["{}{}".format(k, v) for k, v in grouping_vars_folder.items()]))
                                path2plot = path2plot.joinpath(folder_name)
                                Path(path2plot).mkdir(parents=True, exist_ok=True)
                            for grouping_vars, data2plot in group(data2plot_folder,
                                                                  names=vars4plot.union(plot_by, axes_by, folder_by),
                                                                  by=plot_by):
                                # naming the plot
                                extra_info = "_".join(["{}{}".format(k, v) for k, v in grouping_vars.items()])
                                plot_name = (name if name is not None else plot_function.__name__)
                                if len(extra_info) > 1:
                                    plot_name += "_" + extra_info
                                plot_name = clean_str4saving(plot_name)
                                plot_name = f"{path2plot}/{plot_name}{format}"
                                if plot_again or not os.path.exists(plot_name):
                                    yield list(
                                        group(data2plot, names=vars4plot.union(plot_by, axes_by, folder_by),
                                              by=axes_by)), plot_name

                def parallel_func(args):
                    data2plot_per_plot, plot_name = args
                    with timeit("Plot {}\n".format(plot_name)):
                        with many_plots_context(N_subplots=len(data2plot_per_plot), pathname=plot_name, savefig=savefig,
                                                return_fig=True, axes_xy_proportions=axes_xy_proportions,
                                                dpi=dpi) as fax:
                            fig, axes = fax
                            # ylim = tuple()
                            for i, (data_of_ax, data2plot_in_ax) in enumerate(data2plot_per_plot):
                                ax = get_sub_ax(axes, i)
                                ax.set_title("".join(["{}: {}\n".format(k, v) for k, v in data_of_ax.items()]))
                                plot_function(fig=fig, ax=ax,
                                              **{k: v for k, v in data2plot_in_ax.items() if k in function_arg_names},
                                              **extra_arguments)
                                if add_legend:
                                    ax.legend()
                                if xlabel is not None:
                                    ax.set_xlabel(xlabel)
                                if ylabel is not None:
                                    ax.set_ylabel(ylabel)
                            plt.tight_layout()
                            return plot_name

                plot_paths = [plot_name for plot_name in get_map_function(num_cores)(parallel_func, iterator())]
            return plot_paths

        return decorated_func

    return wraper


def unfold(dict_of_lists):
    unfolded_data = {k: [] for k in dict_of_lists.keys()}
    for values in zip(*dict_of_lists.values()):
        lengths = set([len(v) for v in values if not isinstance(v, str) and hasattr(v, "__len__")])
        if len(lengths) > 1:
            raise Exception(f"Can not unfold data with different lengths: {lengths}")
        elif len(lengths) == 1:
            length = lengths.pop()
        else:
            length = 1

        for k, v in zip(dict_of_lists.keys(), values):
            if not isinstance(v, str) and hasattr(v, "__len__"):
                unfolded_data[k] += list(v)
            else:
                unfolded_data[k] += [v] * length

    return unfolded_data


def generic_plot(data_manager: DataManager, x: str, y: str, label: str = None, plot_func: Callable = sns.lineplot,
                 other_plot_funcs=(), log: str = "", sort_by=[], ylim=None, xlim=None, **kwargs):
    # TODO: a way to agregate data instead of splitting depending if sns or plt
    @perplex_plot()
    @with_signature(
        f"plot_{plot_func.__name__}_{y}_vs_{x}_by_{label}(fig, ax, {', '.join({x, y, label}.union(sort_by).difference([None]))})")
    def function_plot(**vars4plot):
        ax = vars4plot["ax"]

        for other_plot in [other_plot_funcs] if isinstance(other_plot_funcs, Callable) else other_plot_funcs:
            other_plot(**{k: vars4plot[k] for k in inspect.getfullargspec(other_plot_funcs).args})

        dict4plot = {
            x: vars4plot[x],
            y: vars4plot[y],
        }
        if label is not None:
            dict4plot[label] = vars4plot[label]
        for var in sort_by:
            dict4plot[var] = vars4plot[var]

        data = pd.DataFrame.from_dict(unfold(dict4plot))
        # data = pd.DataFrame.from_dict(unfold(dict4plot))
        data = data.sort_values(by=sort_by + ([label] if label is not None else []) + [x]).reset_index(drop=True)
        ax.set(xlabel=x, ylabel=y)
        plot_func(data=data, x=x, y=y, hue=label, ax=ax)
        # add proper Dim values as x labels
        # ax.set_xticklabels(data[x])
        # for item in ax.get_xticklabels(): item.set_rotation(90)

        # if "data" in inspect.getfullargspec(plot_func).args:
        #     data = pd.DataFrame.from_dict(
        #         {
        #             x: vars4plot[x],
        #             y: vars4plot[y],
        #             label: vars4plot[label],
        #         }
        #     )
        #     data.sort_values(by=x)
        #     plot_func(data=data, x=x, y=y, hue=label, ax=ax)
        # else:
        #     for x_i, y_i, label_i in zip(vars4plot[x], vars4plot[y], vars4plot[label]):
        #         plot_func(ax, x_i, y_i, label=label_i)
        #     ax.legend()

        if "x" in log:
            ax.set_xscale("log")
        if "y" in log:
            ax.set_yscale("log")

        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)

    return function_plot(data_manager, **kwargs)


def correlation_plot(data_manager: DataManager, axes_var: str, val_1, val_2, value_var, log: str = "", **kwargs):
    @perplex_plot
    @with_signature(
        f"correlation_plot_{axes_var}_{value_var}(fig, ax, {', '.join({axes_var, value_var})})")
    def function_plot(**vars4plot):
        ax = vars4plot["ax"]
        ax.scatter(vars4plot[value_var][0], vars4plot[value_var][1], label="Data")
        ax.plot(vars4plot[value_var][0], vars4plot[value_var][0], ':k', label="Equality line")
        ax.set_xlabel(vars4plot[axes_var][0])
        ax.set_ylabel(vars4plot[axes_var][1])
        ax.legend()

        if "x" in log:
            ax.set_xscale("log")
        if "y" in log:
            ax.set_yscale("log")

    # if we forgot to put some vars add them to the plot_by; but needs to know the arguments of the functions
    # not_specified_vars = set(data_manager.columns).difference(kwargs.get("plot_by", []))
    # not_specified_vars = not_specified_vars.difference(kwargs.get("axes_by", []))
    # not_specified_vars = not_specified_vars.difference(kwargs.keys())
    # not_specified_vars = not_specified_vars.difference([axes_var, value_var])
    # kwargs["plot_by"] = kwargs.get("plot_by", []) + list(not_specified_vars)
    return function_plot(data_manager, **{axes_var: [val_1, val_2]}, **kwargs)


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


def get_sub_ax(ax, i):
    nrows, ncols = ax.shape
    return ax[i // ncols, i % ncols]


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
def save_fig(paths: Union[List, str, Path], filename, show=False, dpi=None):
    yield
    for path in paths if isinstance(paths, list) else [paths]:
        Path(path).mkdir(parents=True, exist_ok=True)
        if "." not in filename:
            filename = f"{filename}.png"
        plt.savefig(f"{path}/{filename}", dpi=dpi)
    if show:
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
