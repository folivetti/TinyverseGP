import sys

from src.analysis.benchmarks.boolean import Conjunction, NegVar
from src.analysis.models.simple_cgp import SimpleCGP, SimpleCGPConfig, MutationType
from src.analysis.problems import MaxPlusMul
from src.gp.tiny_cgp import *
from src.gp.functions import AND

MAX_GENERATIONS = 1000000
MAX_TIME = 9999999
N = 5
MAX_ARITY = 2
NUM_GENES = (MAX_ARITY + 1) * N  + 1
MUTATION_RATE = 1 / NUM_GENES
functions = [AND]
terminals = [Var(i) for i in range(N)] + [NegVar(i) for i in range(N)]

config = SimpleCGPConfig(
    num_jobs=1,
    max_generations=MAX_GENERATIONS,
    stopping_criteria=None,
    minimizing_fitness=True,
    ideal_fitness=None,
    silent_algorithm=False,
    silent_evolver=True,
    minimalistic_output=True,
    num_functions=len(functions),
    max_arity=2,
    num_inputs=len(terminals),
    num_outputs=1,
    report_interval=1,
    max_time=MAX_TIME,
    mutation_type=MutationType.PROB,
    global_seed=None,
    checkpoint_interval=9999999,
    checkpoint_dir='../checkpoint',
    experiment_name='max_tgp'
)

hyperparameters = CGPHyperparameters(
    mu=1,
    lmbda=1,
    population_size=2,
    num_function_nodes=N + 1,
    levels_back= N + 1,
    mutation_rate=MUTATION_RATE,
    strict_selection=False
)


problem = Conjunction(n = N, use_complete_training_set=False)
config.ideal_fitness = problem.ideal
config.global_seed = int(time.time_ns())
cgp = SimpleCGP(functions, terminals, config, hyperparameters)
cgp.evolve(problem)
print(f"{N},simple_cgp,{cgp.generation_number}")
