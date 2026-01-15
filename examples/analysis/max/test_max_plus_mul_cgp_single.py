import sys
from src.analysis.models.simple_cgp import SimpleCGP
from src.analysis.problems import MaxPlusMul
from src.gp.tiny_cgp import *
from src.gp.functions import ADD, MUL
from src.gp.tinyverse import Const

MAX_GENERATIONS = 1000000
MAX_TIME = 9999999
D = int(sys.argv[1])
T = int(sys.argv[2])
functions = [ADD, MUL]
terminals = [Const(T)]

config = CGPConfig(
    num_jobs=1,
    max_generations=MAX_GENERATIONS,
    stopping_criteria=None,
    minimizing_fitness=False,
    ideal_fitness=None,
    silent_algorithm=True,
    silent_evolver=True,
    minimalistic_output=True,
    num_functions=len(functions),
    max_arity=2,
    num_inputs=1,
    num_outputs=1,
    report_interval=1,
    max_time=MAX_TIME,
    global_seed=None,
    checkpoint_interval=10,
    checkpoint_dir='../checkpoint',
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

for _ in range(0,30):
    problem = MaxPlusMul(d=D, t=T)
    config.ideal_fitness = problem.ideal
    config.global_seed = int(time.time_ns())
    cgp = SimpleCGP(functions, terminals, config, hyperparameters)
    cgp.evolve(problem)
    print(f"{D},simple_cgp,{cgp.generation_number}")
