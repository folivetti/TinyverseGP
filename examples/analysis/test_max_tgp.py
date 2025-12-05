from src.analysis.problems import MaxPlusMul, MaxPlus
from src.gp.tiny_cgp import *
from src.gp.functions import ADD, MUL
from src.gp.tiny_tgp import TGPHyperparameters, TinyTGP
from src.gp.tinyverse import Const

D = 4
T = 0.5
MAX_SIZE = math.pow(2, D)
MAX_DEPTH = D + 1

problem = MaxPlus(d=D, t=T)
functions = [ADD]
terminals = [Const(T)]
ideal = problem.ideal

print(ideal)

config = GPConfig(
    num_jobs=1,
    max_generations=10000,
    stopping_criteria=ideal,
    minimizing_fitness=False,
    ideal_fitness=ideal,
    silent_algorithm=False,
    silent_evolver=False,
    minimalistic_output=True,
    num_outputs=1,
    report_interval=100,
    max_time=3600,
    global_seed=None,
    checkpoint_interval=10,
    checkpoint_dir='checkpoint',
    experiment_name='max_tgp'
)

hyperparameters = TGPHyperparameters(
    pop_size=50,
    max_size=MAX_SIZE,
    max_depth=MAX_DEPTH,
    cx_rate=0.9,
    mutation_rate=0.3,
    tournament_size=4,
    penalization_complexity_factor=0.0,
    erc=False
)

tgp = TinyTGP(functions, terminals, config, hyperparameters)
tgp.evolve(problem)
