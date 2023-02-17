"""

# TODO Lab and DataManager should be together?
# TODO: save structure may bi itself another dict with the function as keys and not inside the tuple:
    [subset_dict(input_params, self.variables[variable].root),
    subset_dict(input_funcs, self.variables[variable].dependencies),
    function_block, function_name, variable]
    instead of
    [subset_dict(input_params, self.variables[variable].root),
    subset_dict(input_funcs, self.variables[variable].dependencies),
    variable]
"""

import copy
import itertools
import os
from collections import defaultdict
from logging import warning
from pathlib import Path
from typing import Union, List, Dict, Set

# import h5py
import joblib
from benedict import benedict

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
    def __init__(self, path: Union[str, Path], name: str, format=HD5):
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

        self.database = benedict()

    def get_variable(self, input_params, input_funcs, variable):
        if variable in self.variables.keys():
            return self.database.get([subset_dict(input_params, self.variables[variable].root),
                                      subset_dict(input_funcs, self.variables[variable].dependencies),
                                      variable], None)
        elif variable in self.parameters.keys():
            return input_params[variable]
        elif variable in self.function_blocks.keys():
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
                   function_result: Dict):
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
            self.variables[k].dependencies.update(output_funcs.keys())
            self.variables[k].root.update(input_params.keys())
        # add the result.
        self.database[subset_dict(input_params, input_params.keys()),
        subset_dict(output_funcs, output_funcs.keys())] = function_result

    def __getitem__(self, item):
        if isinstance(item, (set, list)):
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
            everything = set(self.parameters.keys()).union(self.variables.keys()).union(self.function_blocks.keys())
            return self.__getitem__(everything)
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
            joblib.dump((self.database, self.parameters, self.function_blocks, self.variables), self.path_to_data)
        # elif self.format == HD5:
        #     with h5py.File(self.path_to_data, 'w') as f:
        #         f.create_dataset("database", data=self.database)
        #         f.create_dataset("parameters", data=self.parameters)
        #         f.create_dataset("variables", data=self.variables)
        #         f.create_dataset("function_blocks", data=self.function_blocks)
        else:
            raise Exception(f"Data format {self.format} not implemented.")

    def load(self):
        if len(self.database) == 0 and os.path.exists(self.path_to_data):
            if self.format == JOBLIB:
                self.database, self.parameters, self.function_blocks, self.variables = joblib.load(self.path_to_data)
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
