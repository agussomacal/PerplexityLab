import copy
import inspect
import itertools
import os
import warnings
from collections import namedtuple
from contextlib import contextmanager
from inspect import signature
from pathlib import Path
from typing import Callable, List, Union

import joblib
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import seaborn as sns
from makefun import with_signature, wraps

from PerplexityLab.DataManager import DataManager, group, apply, dmfilter, JOBLIB
from PerplexityLab.miscellaneous import timeit, get_map_function, clean_str4saving, filter_dict

INCHES_PER_LETTER = 0.11
INCHES_HEIGHT = 0.2
LEGEND_EXTRA_PERCENTAGE_SPACE = 0.1
DEFAULT_X_AXIS_SIZE = 4
DEFAULT_Y_AXIS_SIZE = 4
DEFAULT_FONT_SIZE = 18


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


def set_latex_fonts(font_family="amssymb", packages=("amsmath",)):
    # ("babel", "lmodern", "amsmath", "amsthm", "amssymb", "amsfonts", "fontenc", "inputenc")
    # preamble=r'\usepackage{babel}\usepackage{lmodern}\usepackage{amsmath,amsthm,amssymb}\usepackage{amsfonts}\usepackage[T1]{fontenc}\usepackage[utf8]{inputenc}'
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": font_family,
    })
    plt.rc('text.latex',
           preamble=r''.join([f"\\usepackage{{{package}}}" for package in packages])
           # )
           )


def get_path_name2replot_data(data_manager, plot_function, name, preimage_format):
    return (f"{data_manager.path}/"
            f"data2replot_{(name if name is not None else plot_function.__name__)}.{preimage_format}")


def get_remove_legend(ax, legend_text, legend_handles, legend_outside_plot):
    if legend_outside_plot is not None:
        # not plot here the legend. seaborn do it automatically so remove it.
        try:
            ax.legend_.remove()
        except AttributeError:
            pass

        legend_text += ax.get_legend_handles_labels()[1]
        legend_handles += ax.get_legend_handles_labels()[0]
        return legend_text, legend_handles


# def get_remove_title():
#     tit = get_name_for_title(sub_statsdata, plot_axes_separator_categories + plot_axes_sort_categories)
#     if isinstance(title, str):
#         tit = title
#     elif isinstance(title, list):
#         tit = tit + '\n' + '\n'.join(title)
#
#     ax.set_title(tit)
#
#     title_len = max(map(len, ax.get_title().split('\n')))
#     title_lines = len(ax.get_title().split('\n'))
#     title_max_len = title_len if title_len > title_max_len else title_max_len
#     title_max_lines = title_lines if title_lines > title_max_lines else title_max_lines

LegendOutsidePlot = namedtuple("LegendOutsidePlot",
                               "loc extra_x_left extra_x_right extra_y_top extra_y_bottom",
                               defaults=["lower center", 0, 0, 0, 0.1])


def plot_legend(fig, legend_text, legend_handles, legend_font_dict, legend_outside_plot: LegendOutsidePlot = None):
    if legend_outside_plot is not None:
        # # calculate how many extra lines to be added for legend
        num_legend_text = len(set(legend_text))
        x0, y0, x, y = np.array(fig.bbox.bounds) / fig.dpi
        fig.set_size_inches((x0 + x) * (1 + legend_outside_plot.extra_x_left + legend_outside_plot.extra_x_right),
                            (y0 + y) * (1 + legend_outside_plot.extra_y_bottom + legend_outside_plot.extra_y_top),
                            forward=True)

        plt.subplots_adjust(
            top=1 - legend_outside_plot.extra_y_top,
            bottom=legend_outside_plot.extra_y_bottom,
            left=legend_outside_plot.extra_x_left,
            right=1 - legend_outside_plot.extra_x_right,
            # hspace=title_max_lines * inches_per_label
        )

        if num_legend_text > 0:
            # plt.subplots_adjust(bottom=num_legend_text * inches_per_label / y)

            # get unique legend text/handlers.
            final_legend_handlers = []
            final_legend_text = []
            for h, t in zip(legend_handles, legend_text):
                if t not in final_legend_text:
                    final_legend_handlers.append(h)
                    final_legend_text.append(t)
            fig.legend(final_legend_handlers, final_legend_text,
                       # bbox_to_anchor=(0.5, 0.1),
                       # loc='lower center' if legend_loc is None else legend_loc,
                       loc=legend_outside_plot.loc,
                       fancybox=True, shadow=False, prop=legend_font_dict)
    else:
        fig.tight_layout()


def perplex_plot(plot_by_default=[], axes_by_default=[], folder_by_default=[], group_by=[], sort_by_default=[],
                 legend=True):
    """

    :param plot_by_default:
    :param axes_by_default:
    :param folder_by_default:
    :param legend:
    :return:
    """
    group_by = group_by if isinstance(group_by, list) else [group_by]

    def wraper(plot_function):
        """

        :param plot_function: plot_function(fig=fig, ax=ax, ...)
        :return:
        """

        def decorated_func(data_manager: DataManager, path=None, name=None, folder="", plot_by=plot_by_default,
                           axes_by=axes_by_default, folder_by=folder_by_default, sort_by=sort_by_default,
                           axes_xy_proportions=(10, 8),
                           savefig=True,
                           axis_font_dict={'color': 'black', 'weight': 'normal', 'size': 14},
                           legend_font_dict={'weight': 'normal', "size": 18, 'stretch': 'normal'},
                           labels_font_dict={'color': 'black', 'weight': 'normal', 'size': 16},
                           xticks=None, yticks=None,
                           font_family="amssymb", title=True,
                           dpi=None, plot_again=True, format=".png", num_cores=1, add_legend=legend, xlabel=None,
                           ylabel=None, usetex=True, create_preimage_data=False, preimage_format=JOBLIB,
                           only_create_preimage_data=False, legend_outside=False, legend_loc=None,
                           legend_outside_plot=None,
                           inches_height=INCHES_HEIGHT, **kwargs):
            format = format if format[0] == "." else "." + format
            if usetex:
                plt.rcParams.update({
                    "text.usetex": usetex,
                    "font.family": font_family,
                })

            # TODO: make it work
            if num_cores > 1:
                warnings.warn("Doesn not work for multiplecores when passing a lambda function.")
            with (data_manager.track_emissions("figures")):

                plot_by = plot_by if isinstance(plot_by, list) else [plot_by]
                axes_by = axes_by if isinstance(axes_by, list) else [axes_by]
                sort_by = sort_by if isinstance(sort_by, list) else [sort_by]
                folder_by = folder_by if isinstance(folder_by, list) else [folder_by]

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

                function_arg_names = inspect.getfullargspec(plot_function).args
                assert len({"fig", "ax"}.intersection(
                    function_arg_names)) == 2, "fig and ax should be two varaibles of ploting " \
                                               "function but they were not found: {}".format(function_arg_names)

                if create_preimage_data and os.path.exists(
                        get_path_name2replot_data(data_manager, plot_function, name, preimage_format)):
                    if preimage_format == JOBLIB:
                        dm, vars4plot, names4plot, specified_vars, extra_arguments = \
                            joblib.load(get_path_name2replot_data(data_manager, plot_function, name, preimage_format))
                    else:
                        raise Exception("Not implemented yet.")
                else:
                    vars4plot = set(function_arg_names).intersection(data_manager.columns)
                    specified_vars = {
                        k: v if isinstance(v, list) else [v] for k, v in kwargs.items() if
                        k in data_manager.columns
                    }
                    functions2apply = {
                        k: v for k, v in kwargs.items() if
                        k in function_arg_names + plot_by + axes_by + folder_by + group_by + sort_by
                        and k not in specified_vars.keys() and isinstance(v, Callable)
                        and set(inspect.getfullargspec(v).args).issubset(data_manager.columns)
                    }
                    functions2apply_var_needs = set(itertools.chain(*[
                        inspect.getfullargspec(v).args for k, v in kwargs.items() if
                        k in function_arg_names + plot_by + axes_by + folder_by + group_by + sort_by
                        and k not in specified_vars.keys() and isinstance(v, Callable)
                        and set(inspect.getfullargspec(v).args).issubset(data_manager.columns)
                    ]))
                    extra_arguments = {k: v for k, v in kwargs.items() if
                                       k in function_arg_names and k not in specified_vars.keys()
                                       and k not in functions2apply.keys()}
                    names = vars4plot.union(plot_by, axes_by, folder_by, group_by, sort_by,
                                            specified_vars.keys()).difference(functions2apply.keys())
                    dm = dmfilter(data_manager, names.union(functions2apply_var_needs),
                                  **specified_vars)  # filter first by specified_vars
                    dm = apply(dm, names=names, **functions2apply)  # now apply the functions
                    vars4plot.update(functions2apply.keys())
                    names4plot = vars4plot.union(plot_by, axes_by, folder_by, group_by, sort_by)

                    if create_preimage_data:
                        if preimage_format == JOBLIB:
                            joblib.dump((dm, vars4plot, names4plot, specified_vars, extra_arguments),
                                        get_path_name2replot_data(data_manager, plot_function, name, preimage_format))
                        else:
                            raise Exception("Not implemented yet.")

                if only_create_preimage_data:
                    return []
                else:
                    def iterator():
                        for grouping_vars_folder, data2plot_folder in \
                                group(dm, names=names4plot, by=folder_by,
                                      **specified_vars):
                            for path2plot in paths:
                                if len(folder_by) > 0:
                                    folder_name = clean_str4saving(
                                        "_".join(["{}{}".format(k, v) for k, v in grouping_vars_folder.items()]))
                                    path2plot = path2plot.joinpath(folder_name)
                                    Path(path2plot).mkdir(parents=True, exist_ok=True)
                                for grouping_vars, data2plot in group(data2plot_folder,
                                                                      names=names4plot,
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
                                            group(data2plot, names=names4plot,
                                                  by=axes_by, sort_by=sort_by)), plot_name

                    def parallel_func(args):
                        data2plot_per_plot, plot_name = args
                        with timeit("Plot {}\n".format(plot_name)):
                            with many_plots_context(N_subplots=len(data2plot_per_plot), pathname=plot_name,
                                                    savefig=savefig,
                                                    return_fig=True, axes_xy_proportions=axes_xy_proportions,
                                                    dpi=dpi) as fax:
                                legend_text = []
                                legend_handles = []
                                fig, axes = fax
                                for i, (data_of_ax, data2plot_in_ax) in enumerate(data2plot_per_plot):
                                    ax = get_sub_ax(axes, i)
                                    if title:
                                        ax.set_title("".join(["{}: {}\n".format(k, v) for k, v in data_of_ax.items()]),
                                                     fontdict=legend_font_dict)
                                    for selection, sub_data2plot_in_ax in group(data2plot_in_ax, names=names4plot,
                                                                                by=group_by, sort_by=sort_by):
                                        plot_function(fig=fig, ax=ax,
                                                      **{k: v if k not in group_by else selection[k] for k, v in
                                                         sub_data2plot_in_ax.items() if
                                                         k in function_arg_names},
                                                      **extra_arguments)
                                    if add_legend:
                                        ax.legend(prop=legend_font_dict)
                                        get_remove_legend(ax, legend_text, legend_handles,
                                                          legend_outside_plot=legend_outside_plot)
                                    if xlabel is not None:
                                        ax.set_xlabel(xlabel, fontdict=labels_font_dict)
                                    if ylabel is not None:
                                        ax.set_ylabel(ylabel, fontdict=labels_font_dict)

                                    if xticks is not None:
                                        ax.set_xticks(xticks, xticks)
                                    if yticks is not None:
                                        ax.set_yticks(yticks, yticks)

                                    # take the same size as the label
                                    if "size" in axis_font_dict.keys():
                                        ax.tick_params(
                                            labelsize=axis_font_dict[
                                                "size"] if "size" in axis_font_dict.keys() else None)
                                plot_legend(fig, legend_text, legend_handles, legend_font_dict,
                                            legend_outside_plot=legend_outside_plot)
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
        data = data.sort_values(by=sort_by + ([label] if label is not None else []) + [x]).reset_index(
            drop=True)
        # ax.set(xlabel=x, ylabel=y, fontdict=kwargs["axis_font_dict"])
        ax.set_xlabel(x, fontdict=kwargs["labels_font_dict"])
        ax.set_ylabel(y, fontdict=kwargs["labels_font_dict"])

        plot_func(data=data, x=x, y=y, hue=label, ax=ax)

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
