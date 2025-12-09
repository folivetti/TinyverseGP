import numpy as np

from src.analysis.models.simple_cgp import SimpleCGP
from src.analysis.problems import MaxPlusMul
from src.gp.tiny_cgp import *
from src.gp.functions import ADD, MUL
from src.gp.tinyverse import Const

NUM_JOBS = 10
D = 4
T = 1
problem = MaxPlusMul(d=D, t=T)
functions = [ADD, MUL]
terminals = [Const(T), Const(0)]
ideal = problem.ideal

print(ideal)

config = CGPConfig(
    num_jobs=100,
    max_generations=5000000,
    stopping_criteria=ideal,
    minimizing_fitness=False,
    ideal_fitness=ideal,
    silent_algorithm=True,
    silent_evolver=False,
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
    mutation_rate=1.0/D,
    strict_selection=False,
)

evals = []
for _ in range(NUM_JOBS):
    cgp = SimpleCGP(functions, terminals, config, hyperparameters)
    best = cgp.evolve(problem)
    evals.append(cgp.num_evaluations)

print(np.mean(evals))