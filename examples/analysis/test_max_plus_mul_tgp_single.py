from src.analysis.problems import MaxPlusMul
from src.gp.tiny_cgp import *
from src.gp.functions import ADD, MUL
from src.gp.tiny_tgp import TGPConfig
from src.gp.tinyverse import Const
from src.analysis.models.simple_tgp import SimpleTGP, SGPHyperparameters

MAX_GENERATIONS = 5000000
D = 7
T = 1
functions = [ADD, MUL]
terminals = [Const(T)]

config = TGPConfig(
    num_jobs=1,
    max_generations=MAX_GENERATIONS,
    stopping_criteria=None,
    minimizing_fitness=False,
    ideal_fitness=None,
    silent_algorithm=True,
    silent_evolver=True,
    minimalistic_output=True,
    num_outputs=1,
    report_interval=1,
    max_time=3600,
    global_seed=None,
    checkpoint_interval=10,
    checkpoint_dir='checkpoint',
    experiment_name='max_tgp'
)

hyperparameters = SGPHyperparameters(
    lmbda=1,
    k=1,
    strict_selection = False
)

problem = MaxPlusMul(d=D, t=T)
config.ideal_fitness = problem.ideal
config.global_seed = int(time.time_ns())
tgp = SimpleTGP(functions, terminals, config, hyperparameters)
tgp.evolve(problem)

print(f"{D};simple_tgp;{tgp.num_evaluations}")