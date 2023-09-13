# <font color="navy"> Perplexity</font><font color="green">Lab</font>

— *Hi! Listen! I need your help!<br />
Cause' I'm bored again to hell<br />
with the prospect of re-doing,<br />
for this project, the annoying<br />
repeated and painful coding!* 

— *I am here for you,<br /> 
tell me dear friend,<br /> 
what's your conundrum?*

— *What I would really like is vast, <br />
can some pip install magic path <br />
give my code what it needs to have?*
* __Reproducible research__: everyone everywhere should be able to run my code and get the same results. And the main script should be kind of easy to read.
* __File management__: I don't want to waste my time in deciding where to store and get the data and the results.
* __Explore multiple conditions__: I want to execute my experiments/simulations under different conditions specified by my relevant parameters.
* __Parallel__: I want to run several experiments/simulations in parallel without having to rewrite code to do it specifically each time.
* __Remember__: what has been done without worrying about the format nor the place. So I can come later in a year and do old or new analysis and don't straggle with forgotten files and paths. 
* __Avoid re-doing__: automatically check if some experiment was done and load instead of re-doing.
* __Analysis__: once experimentation is done I'd wish to produce with minimal coding some generics or customized plots to analyse the results and get insight on the project.
* __Make reports__: Connect results directly to latex to create a report on the fly. Or directly modify the plots that will be presented without any troublesome file management.
* __Explorable reports__: Create jupyter notebooks with widgets to explore or show results interactively.
* __Carbon and energy footprint__: and why not to get with zero extra effort the accumulated amount of equivalent CO2 and energy consumption of my research to be more conscious and work towards more environmentally responsible research?

— Dear friend, what you need is __PerplexityLab__!
Pipelines Experiments Reproducible Parallel Latex Environmentally conscIous jupYter widgets... Or something around that lines. Anyway, it does that and more! Give it a try!

So just to give you a flavour, but in truth you can do much more!

For a more detailed explanation:
* [PerplexityLab for Machine Learning](https://github.com/agussomacal/PerplexityLab/blob/main/src/examples/PerplexityLab4ML.ipynb)

To see real projects using PerplexityLab:
* [Non Linear Compressive Reduced Basis Approximation for PDE’s](https://github.com/agussomacal/NonLinearRBA4PDEs)
 

## Installation

``` bash
pip install -i https://test.pypi.org/simple/ PerplexityLab==0.0.1b0
```

## How does it work?

First define your experiment as functions whose output should be a dictionary with string keys. 
``` python
def experiment1(variable1, variable2):
    # do something here
    return {
        "result1": result1
        "result2": result2    
    }
```
Or set of experiments. The names of the input and output variables are relevant because they can be the input to new experimental layers or later for doing analysis.
``` python
def experiment2(variable1, result2):
    # do another something here
    return {
        "result3": result3    
    }
```

Define a __*DataManager*__ giving a __path__ (if not existent it will be created) where results will be stored and a __name__.
Optionally you can add a storing format (default is joblib), and set __trackCO2__ to True (default False) if we want to track the CO2 
emissions and energy consumption each time our experiments are executed.
``` python
dm = DataManager(
    path="~/Somewhere/Maybe/Here",
    name="MyLaboratory", #All the data and results will be in "~/Somewhere/Maybe/Here/MyLaboratory/"
    format=JOBLIB,
    trackCO2=True
)
```

Create a __*LabPipeline*__ instance and add your experiment layers specifying a name for the layer. Multiple functions can be associated with the same layer:
``` python
lab = LabPipeline()
lab.define_new_block_of_functions("preprocessing", experiment1)
lab.define_new_block_of_functions("experiment", experiment2)
```

Ready? Run your experiment! You just need to specify
- the *DataManager* object that will be use to retrieve old experiments
- the number of cores to run in parallel
- if we wish to re-do the experiments even if they have been already computed.
- maybe we wish to start anew and forget everything.

Finally, specify a list of values each input variable can take. The experiment functions defined above will be executed for each combination of the input variables (cartesian product).
``` python
dm = lab.execute(
    data_manager=dm, 
    num_cores=1, 
    recalculate=False,
    forget=False, 
    variable1=["fantastic", "easy"], 
    variable2=[0, 1, 10]
)
```

Now the analysis. Some already ready plots can be used without any extra work.
For example, to see the behaviour of *result2* as a function of *variable2* we just
call to generic_plot and pass the variable names to the x and y parameters. We can add
a label following a third variable and decide to split the plot in several axes or plots 
with axes_by and plot_by. You can also create new variables on the fly and use them
as label or y or x variable. And you can filter some variables to only plot the
specified subset of them by simply saying which list of possible values should be shown.
``` python
generic_plot(
    data_manager=dm, 
    x="variable2", 
    y="result2", 
    label="newvar", 
    plot_func=sns.lineplot,
    newvar=lambda variable1: variable1+" indeed", 
    variable1=["fantastic"],
    axes_by=["result3"],
    plot_by=[]    
)
```

But if you need more complex plots and still keep the aforementioned facilities, just
add a decorator to your definition.
``` python
@perplex_plot
def my_plot(fig, ax, variable1, result1, **kwargs):
    #do something
```
And then we can just call:
``` python
my_plot(
    data_manager=dm,
    variable1=["fantastic"],
    axes_by=["result3"],
    plot_by=["result1"]    
)
```


Coming soon:
* Create jupyter with widgets using [Quarto](https://quarto.org/)
* Solve issue when input vars of functions in same layer differ
* Add example with physics simulations
* Add example with experimental physics analysis

and eventually
* Logo
* Experiments whose output is a file saved in memory.
* Database made of numerical keys mapped to list of stored variables (instead of te variables themselves being the keys and values).
* Layers not to be saved (computed every time).
* Each Layer saved in different files.
* Forget selected experiments.
* Variables that are actualized after another computation should replace the old ones?
* Lab and DataManager should be together?
* Input Variables not stringables.
* Default variables in experiment functions should be accepted even if we don't add them as input vars explicitly.

