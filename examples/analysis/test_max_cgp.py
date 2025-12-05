from src.analysis.problems import Max
from src.gp.tiny_cgp import *
from src.gp.problem import BlackBox
from src.gp.functions import ADD, MUL
from src.gp.tinyverse import Const

D = 5
IDEAL = (4**2)**(D - 3)
functions = [ADD, MUL]
terminals = [Const(0.5), Const(0)]

config = CGPConfig(
    num_jobs=1,
    max_generations=10000,
    stopping_criteria=IDEAL,
    minimizing_fitness=False,
    ideal_fitness=IDEAL,
    silent_algorithm=False,
    silent_evolver=False,
    minimalistic_output=True,
    num_functions=len(functions),
    max_arity=2,
    num_inputs=2,
    num_outputs=1,
    report_interval=1,
    max_time=60,
    global_seed=42,
    checkpoint_interval=10,
    checkpoint_dir='checkpoint',
    experiment_name='sr_cgp'
)

hyperparameters = CGPHyperparameters(
    mu=1,
    lmbda=1,
    population_size=2,
    num_function_nodes=D,
    levels_back=99999,
    mutation_rate=0.1,
    strict_selection=False,
)

problem = Max(d_=5)
cgp = TinyCGP(functions, terminals, config, hyperparameters)
cgp.evolve(problem)
