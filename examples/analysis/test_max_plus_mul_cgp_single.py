import seaborn as sns
from src.analysis.models.simple_cgp import SimpleCGP
from src.analysis.problems import MaxPlusMul
from src.gp.tiny_cgp import *
from src.gp.functions import ADD, MUL
from src.gp.tinyverse import Const

MAX_GENERATIONS = 5000000
D = 4
T = 1
functions = [ADD, MUL]
terminals = [Const(T), Const(0)]

config = CGPConfig(
    num_jobs=1,
    max_generations=5000000,
    stopping_criteria=None,
    minimizing_fitness=False,
    ideal_fitness=None,
    silent_algorithm=True,
    silent_evolver=True,
    minimalistic_output=True,
    num_functions=len(functions),
    max_arity=2,
    num_inputs=2,
    num_outputs=1,
    report_interval=1,
    max_time=3600,
    global_seed=None,
    checkpoint_interval=10,
    checkpoint_dir='checkpoint',
    experiment_name='max_tgp'
)

hyperparameters = CGPHyperparameters(
    mu=1,
    lmbda=1,
    population_size=2,
    num_function_nodes=D,
    levels_back=D,
    mutation_rate=None,
    strict_selection=False,
)

problem = MaxPlusMul(d=D, t=T)
config.ideal_fitness = problem.ideal
config.global_seed = int(time.time_ns())
cgp = SimpleCGP(functions, terminals, config, hyperparameters)
cgp.evolve(problem)

print(f"{D};simple_cgp;{cgp.num_evaluations}")
