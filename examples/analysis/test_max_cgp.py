import math

from src.analysis.problems import Max
from src.gp.tiny_cgp import *
from src.gp.functions import ADD, MUL
from src.gp.tinyverse import Const

D = 8
T = 1
problem = Max(d=D, t=T)
functions = [ADD, MUL]
terminals = [Const(T), Const(0)]
ideal = problem.ideal

config = CGPConfig(
    num_jobs=1,
    max_generations=10000,
    stopping_criteria=ideal,
    minimizing_fitness=False,
    ideal_fitness=ideal,
    silent_algorithm=True,
    silent_evolver=False,
    minimalistic_output=True,
    num_functions=len(functions),
    max_arity=2,
    num_inputs=1,
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
    num_function_nodes=D-1,
    levels_back=D,
    mutation_rate=0.1,
    strict_selection=False,
)

cgp = TinyCGP(functions, terminals, config, hyperparameters)
best = cgp.evolve(problem)
