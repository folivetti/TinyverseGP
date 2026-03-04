import sys

from src.analysis.benchmarks.boolean import Conjunction, NegVar
from src.analysis.models.simple_tgp import SimpleTGPHyperparameters, SimpleTGP
from src.gp.tiny_cgp import *
from src.gp.functions import AND
from src.gp.tiny_tgp import TGPConfig

MAX_GENERATIONS = 1000000
MAX_TIME = 9999999
N = 5
functions = [AND]
terminals = [Var(i) for i in range(N)] + [NegVar(i) for i in range(N)]

config = TGPConfig(
    num_jobs=1,
    max_generations=MAX_GENERATIONS,
    stopping_criteria=0,
    minimizing_fitness=True,
    ideal_fitness=0,
    silent_algorithm=False,
    silent_evolver=True,
    minimalistic_output=True,
    num_outputs=1,
    report_interval=1000,
    max_time=MAX_TIME,
    global_seed=None,
    checkpoint_interval=9999999,
    checkpoint_dir='../checkpoint',
    experiment_name='max_tgp'
)

hyperparameters = SimpleTGPHyperparameters(
    lmbda=1,
    k=1,
    strict_selection=False,
    check_size=False,
    max_depth=N*N,
    multi=True
)

if hyperparameters.multi:
    appendix = "multi"
else:
    appendix = "single"


problem = Conjunction(n = N, use_complete_training_set=True)
config.ideal_fitness = problem.ideal
config.global_seed = int(time.time_ns())
tgp = SimpleTGP(functions, terminals, config, hyperparameters)
tgp.evolve(problem)

print(f"{N},simple_tgp_{appendix},{tgp.generation_number}")
