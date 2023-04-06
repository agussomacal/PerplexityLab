import inspect
from collections import namedtuple, OrderedDict
from typing import Callable, Union

from tqdm import tqdm

from src.DataManager import DataManager, experiment_param_generator, common_ancestors
from src.performance_utils import get_map_function

FunctionBlock = namedtuple("LayerFunction", "name function")
InputToParallel = namedtuple("InputToParallel", "input_params input_funcs input_vars function function_name")


class LabPipeline:
    def __init__(self):
        self.experimental_graph = []

    def define_new_block_of_functions(self, name, *functions: Union[FunctionBlock, Callable], **kwargs):
        """
        :param name: name of the layer
        :param functions: tuples of name and corresponding function that will be executed in parallel
        (independently) at this level.
        :return:
        """
        save = True if "save" not in kwargs.keys() else kwargs["save"]
        if all([isinstance(function, Callable) for function in functions]):
            functions = [FunctionBlock(name=f.__name__, function=f) for f in functions]

        assert all([isinstance(function, FunctionBlock) for function in functions]), \
            "functions should be type LayerFunction."
        assert all([isinstance(function.name, str) for function in functions]), "function name should be str."
        assert all([isinstance(function.function, Callable) for function in functions]), \
            "function name should be Callable."
        self.experimental_graph.append((name, save, functions))

    def execute(self, datamanager: DataManager, num_cores=1, forget=False, recalculate=False, save_on_iteration=None,
                verbose=0, **params):
        if forget:
            datamanager.reset()
        else:
            datamanager.load()

        for function_block, save, functions in self.experimental_graph:

            # Generator to avoid storing in memory the unfolded data which could contain big duplicated variables.
            def input_generator():
                for function_name, function in functions:
                    function_arg_names = inspect.getfullargspec(function).args
                    # search through the dependencies which is the minimal set of root parameters that needs to be
                    # explored through cartesian product to call the function and search for the corresponding dependant
                    # variables already present in the dataset.

                    # params mentioned directly in the function arguments call
                    function_param_ancestors = set(function_arg_names).intersection(params.keys())
                    # params obtained through dependencies of function_block names needed for the function call
                    function_param_ancestors.update(
                        common_ancestors(function_arg_names, datamanager.function_blocks.copy(), by="root"))
                    # params obtained through dependencies of variable names needed for the function call
                    function_param_ancestors.update(
                        common_ancestors(function_arg_names, datamanager.variables.copy(), by="root"))
                    function_param_ancestors = sorted(list(function_param_ancestors))

                    # 1) Input general params,
                    # Only iterates on the params that the function needs: for direct call or for dependency with
                    # other variables.
                    for input_params in experiment_param_generator(
                            **OrderedDict([(k, params[k]) for k in function_param_ancestors])):

                        # 2) Specific params needed for the evaluation of the function: this can be another variable
                        # that was created afterwards in a previous layer. (A, B) -> (C) and f(A, C) needs A and C
                        # so the roots are A and B but it needs for evaluation A and C.
                        # If in a previous layer there where many functions, all of this have to be tested if the
                        # needed variable is the output of each of this previous functions.
                        for input_funcs, input_vars in datamanager.experiments_iterator(input_params,
                                                                                        function_arg_names):
                            # checks if it is already done or not
                            if recalculate or not datamanager.is_in_database(input_params, input_funcs,
                                                                             function_block, function_name):
                                # 3) The actual function to be evaluated and the corresponding name.
                                yield InputToParallel(input_params=input_params,
                                                      input_funcs=input_funcs,
                                                      input_vars=input_vars,
                                                      function_name=function_name,
                                                      function=function)

            def parallel_func(vars: InputToParallel):
                result = vars.function(**vars.input_vars)
                return vars.input_params, vars.input_funcs, vars.function_name, result

            with datamanager.track_emissions(function_block):
                input_list = list(input_generator())
                if len(input_list) > 0:
                    for i, (input_params, input_funcs, f_name, f_result) in tqdm(enumerate(
                            get_map_function(num_cores)(parallel_func, input_list)),
                            desc="Doing {}...".format(function_block)):
                        datamanager.add_result(input_params, input_funcs, function_block, f_name, f_result, save)
                        # save after each result only if certain iterations passed
                        if save_on_iteration is not None and save_on_iteration > 0 \
                                and (i % save_on_iteration) == (-1 % save_on_iteration):
                            datamanager.save()

                    # save after each layer
                    if save_on_iteration is None or save_on_iteration > 0:
                        datamanager.save()
                else:
                    print("\r Experiments for {} already done, skipping.".format(function_block))

        return datamanager
