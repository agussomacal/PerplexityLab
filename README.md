# PerplexityLab
Pipelines Experiments Reproducible Parallel Latex reports jupYter widgets

### To Do list
* Layers not to be saved (computed every time)
* Variables that are actualized after another computation should replace the old ones?
* Lab and DataManager should be together?
* save structure may bi itself another dict with the function as keys and not inside the tuple:
    [subset_dict(input_params, self.variables[variable].root),
    subset_dict(input_funcs, self.variables[variable].dependencies),
    function_block, function_name, variable]
    instead of
    [subset_dict(input_params, self.variables[variable].root),
    subset_dict(input_funcs, self.variables[variable].dependencies),
    variable]