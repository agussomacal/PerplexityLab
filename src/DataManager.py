import copy
import inspect
import itertools
import os
from collections import defaultdict, OrderedDict
from logging import warning
from pathlib import Path
from typing import Union, List, Dict, Set, Tuple, Callable, Generator

# import h5py
import joblib
from benedict import benedict

from src.performance_utils import timeit

ALL = "AllParamsBlockVars"

JOBLIB = "joblib"
PICKLE = "pickle"
HD5 = "hd5"
OPTIMIZE_PER_TYPE = "optimize"
VALID_FORMATS = [JOBLIB, PICKLE, HD5, OPTIMIZE_PER_TYPE]


class DatasetParam:
    def __init__(self):
        self.root = set()
        self.values = set()


class DatasetVar:
    def __init__(self):
        self.root = set()
        self.dependencies = set()


class DatasetFBlock(DatasetVar):
    def __init__(self):
        super().__init__()
        self.functions = set()


def experiment_param_generator(**kwargs):
    for values in itertools.product(*list(kwargs.values())):
        yield dict(zip(kwargs.keys(), copy.deepcopy(values)))


def subset_dict(dictionary, keys) -> List:
    """

    :param dictionary:
    :param keys:
    :return: tuple of keys and tuple of values sorted by key names
    """
    return list(map(str, zip(*sorted([(k, dictionary[k]) for k in keys]))))


def common_ancestors(names, data: Dict[str, Union[DatasetParam, DatasetFBlock, DatasetVar]], by="dependencies") -> Set:
    return set().union(itertools.chain(*[getattr(data[name], by) for name in names]))


class DataManager:
    def __init__(self, path: Union[str, Path], name: str, format=JOBLIB):
        self.name = name
        self.path = Path(path).joinpath(name)
        self.path.mkdir(parents=True, exist_ok=True)

        if format not in VALID_FORMATS:
            raise Exception(f"Saving format {format} not implemented. Valid formats are: {VALID_FORMATS}.")
        self.format = format

        self.path_to_data = Path.joinpath(self.path, f"data.{format}")

        self.parameters = defaultdict(DatasetParam)
        self.function_blocks = defaultdict(DatasetFBlock)
        self.variables = defaultdict(DatasetVar)

        self.not_save_vars = set()

        self.database = benedict()

    @property
    def columns(self):
        return set(self.parameters.keys()).union(self.variables.keys()).union(self.function_blocks.keys())

    def get_variable(self, input_params, input_funcs, variable):
        if variable in self.variables.keys():
            return self.database.get([subset_dict(input_params, self.variables[variable].root),
                                      subset_dict(input_funcs, self.variables[variable].dependencies),
                                      variable], None)
        elif variable in input_params.keys():
            return input_params[variable]
        elif variable in input_funcs.keys():
            return input_funcs[variable]
        else:
            raise Exception(f"Variable {variable} not in dataset.")

    def get_output_root_and_vars(self, input_params, input_funcs, function_block):
        # TODO: this can be saved instead of being recalculated
        # the new vars to be considered will depend on the previous computed blocks/funcs and the actual block/func
        output_vars_dependencies = set(input_funcs.keys()).union([function_block])
        output_vars_root = common_ancestors(output_vars_dependencies, self.function_blocks.copy(), by="root")
        output_vars_root.update(input_params.keys())
        return output_vars_root, output_vars_dependencies

    def is_in_database(self, input_params, input_funcs=None, function_block=None, function_name=None):
        if input_funcs is None:
            return [subset_dict(input_params, input_params.keys())] in self.database
        elif function_block is not None and function_name is not None:
            output_funcs = {**input_funcs, **{function_block: function_name}}
            return [subset_dict(input_params, input_params.keys()),
                    subset_dict(output_funcs, output_funcs.keys())] \
                in self.database
        else:
            raise Exception("is_in_database not implemented for that combination of Nones")

    def experiments_iterator(self, input_params, variables):
        dependencies = common_ancestors(variables, self.variables.copy(), by="dependencies")
        dependencies.update(common_ancestors(variables, self.function_blocks.copy(), by="dependencies"))
        if len(dependencies) > 0:
            for input_funcs in experiment_param_generator(
                    **{fb: self.function_blocks[fb].functions for fb in dependencies}):
                yield input_funcs, {variable: self.get_variable(input_params, input_funcs, variable) for variable in
                                    variables}
        else:
            # yield dict(), input_params  # should be equivalent
            yield dict(), {variable: input_params[variable] for variable in variables}

    def add_result(self, input_params: Dict, input_funcs: Dict, function_block: str, function_name: str,
                   function_result: Dict, save=True):
        """
        Add new (or override old) result and actualize information on variables not previously taken into account.
        Here the tree structure of the database is defined.
        :param function_block:
        :param input_funcs:
        :param input_params: the params the function actually needs (where previously filtered in the execution process)
        :param function_name:
        :param function_result:
        :return:
        """
        # add parameters to datamanager if they weren't there before.
        for k, v in input_params.items():
            self.parameters[k].values.add(v)
            self.parameters[k].root.update(input_params.keys())
        # add the function to the block if it was not yet.
        # the actual block and function is in the dependencies (self-depends)
        output_funcs = {**input_funcs, **{function_block: function_name}}
        self.function_blocks[function_block].functions.add(function_name)
        self.function_blocks[function_block].root.update(input_params.keys())
        self.function_blocks[function_block].dependencies.update(output_funcs.keys())
        # add the variables dependencies.
        for k, v in function_result.items():
            if not save:
                self.not_save_vars.add(k)
            self.variables[k].dependencies.update(output_funcs.keys())
            self.variables[k].root.update(input_params.keys())
        # add the result.
        self.database[subset_dict(input_params, input_params.keys()),
        subset_dict(output_funcs, output_funcs.keys())] = function_result

    def __getitem__(self, item):
        if isinstance(item, (set, list, tuple)):
            result_dict = {k: [] for k in item}  # already fixed the output dict keys given by the name sin item.

            common_params = set(item).intersection(self.parameters.keys())
            common_params.update(common_ancestors(item, self.parameters.copy(), by="root"))
            common_params.update(common_ancestors(item, self.variables.copy(), by="root"))
            common_params.update(common_ancestors(item, self.function_blocks.copy(), by="root"))

            for input_params in experiment_param_generator(**{p: self.parameters[p].values for p in common_params}):
                if self.is_in_database(input_params):
                    for input_funcs, selected_variables in self.experiments_iterator(input_params, item):
                        for k, v in selected_variables.items():
                            result_dict[k].append(v)
            return result_dict
        elif item == ALL:
            return self.__getitem__(self.columns)
        elif isinstance(item, str):
            # one only item gives the list directly instead of the dictionary, for that should be given in list form
            # for example ["name"]
            return list(self.__getitem__([item])[item])
        else:
            raise Exception("Not implemented.")

    # ---------- database (save/load/modify) ----------
    def reset(self):
        self.database = benedict()

    def save(self):
        if self.format == JOBLIB:
            joblib.dump((
                # self.database.flatten("/").filter(lambda k, v: k.split("/")[-1] not in self.not_save_vars).unflatten(),
                self.database,
                self.parameters, self.function_blocks, self.variables),
                self.path_to_data)
        # elif self.format == HD5:
        #     with h5py.File(self.path_to_data, 'w') as f:
        #         f.create_dataset("database", data=self.database)
        #         f.create_dataset("parameters", data=self.parameters)
        #         f.create_dataset("variables", data=self.variables)
        #         f.create_dataset("function_blocks", data=self.function_blocks)
        else:
            raise Exception(f"Data format {self.format} not implemented.")

    def load(self):
        with timeit("Loading dataset"):
            if len(self.database) == 0 and os.path.exists(self.path_to_data):
                if self.format == JOBLIB:
                    self.database, self.parameters, self.function_blocks, self.variables = joblib.load(
                        self.path_to_data)
                # elif self.format == HD5:
                #     with h5py.File(self.path_to_data, 'r') as f:
                #         self.database = f['database']
                #         self.parameters = f['parameters']
                #         self.variables = f['variables']
                #         self.function_blocks = f['function_blocks']
                else:
                    raise Exception(f"Data format {self.format} not implemented.")
            else:
                warning(f"Data file in {self.path_to_data} not found.")


# =========== =========== =========== #
#         Other useful function       #
# =========== =========== =========== #
def get_sub_dataset(datamanager: Union[DataManager, Dict[str, List]], names: Union[List, Set]) -> Dict[str, List]:
    assert isinstance(names, (set, list)), f"names should be a list or set of names even if it is only one."
    if isinstance(datamanager, DataManager):
        sub_dataset = datamanager[names]
    elif isinstance(datamanager, dict):
        sub_dataset = {k: datamanager[k] for k in names}  # filter by names
    else:
        raise Exception("Not implemented grouping for type {}".format(type(datamanager)))
    return sub_dataset


def dmfilter(datamanager: Union[DataManager, Dict[str, List]], names: Union[List, Set], **kwargs: List):
    sub_dataset = get_sub_dataset(datamanager, names=set(names).union(kwargs.keys()))
    length = len(list(sub_dataset.values())[0])
    accepted_indexes = [i for i in range(length) if
                        all([sub_dataset[k][i] in (v if isinstance(v, list) else [v]) for k, v in kwargs.items()])]
    return {k: [sub_dataset[k][i] for i in accepted_indexes] for k in names}


def group(datamanager: Union[DataManager, Dict[str, List]], names: Union[List, Set], by: Union[Set, List],
          **filters: List) \
        -> Generator[Tuple[Dict[str, List], Dict[str, List]], None, None]:
    assert isinstance(by, (set, list)), f"by should be a list or set of names even if it is only one."
    sub_dataset = dmfilter(datamanager, names=set(names).union(by), **filters)

    if len(by) > 0:
        length = len(sub_dataset[next(iter(by))])
        order = sorted(zip(range(length), map(str, zip(*[sub_dataset[k] for k in by]))), key=lambda x: x[1])
        for _, indexes in itertools.groupby(order, key=lambda x: x[1]):
            indexes = [i[0] for i in indexes]
            yield OrderedDict([(k, sub_dataset[k][indexes[0]]) for k in by]), \
                OrderedDict([(k, [sub_dataset[k][i] for i in indexes]) for k in names])
    else:
        yield dict(), OrderedDict([(k, sub_dataset[k]) for k in names])


def apply(datamanager: Union[DataManager, Dict], names: Union[Set[str], List[str]], **kwargs: Callable):
    """

    :param datamanager:
    :param names: variables to be included forcefully even if they are not used in the functions to be applied.
    :param kwargs:
    :return:
    """
    assert all(map(lambda x: isinstance(x, Callable), kwargs.values())), "all **kwargs should be callables."
    variables_for_callables = set(
        itertools.chain(*[inspect.getfullargspec(function).args for function in kwargs.values()]))
    sub_dataset = get_sub_dataset(datamanager, variables_for_callables.union(names))
    length = len(list(sub_dataset.values())[0])
    out_dict = dict()
    for function_name, function in kwargs.items():
        # TODO: use groupby to do the implementation more efficient and only be called once
        input_vars_names = inspect.getfullargspec(function).args
        out_dict.update(
            {function_name: [function(**{k: sub_dataset[k][i] for k in input_vars_names}) for i in range(length)]}
        )
    out_dict.update({k: sub_dataset[k] for k in names})
    return out_dict
