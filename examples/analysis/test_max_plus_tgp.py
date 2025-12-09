from time import sleep

import numpy as np

from src.analysis.problems import MaxPlusMul, MaxPlus
from src.gp.tiny_cgp import *
from src.gp.functions import ADD, MUL
from src.gp.tiny_tgp import TGPHyperparameters, TinyTGP
from src.gp.tinyverse import Const

NUM_INSTANCES = 10
D = 3
T = 1
MAX_SIZE = math.pow(2, D + 1)
MAX_DEPTH = D + 1

problem = MaxPlus(d=D, t=T)
functions = [ADD]
terminals = [Const(T)]
ideal = problem.ideal

print(f"Maximum value: {ideal}")

config = GPConfig(
    num_jobs=1,
    max_generations=100000,
    stopping_criteria=ideal,
    minimizing_fitness=False,
    ideal_fitness=ideal,
    silent_algorithm=True,
    silent_evolver=False,
    minimalistic_output=True,
    num_outputs=1,
    report_interval=100,
    max_time=3600,
    global_seed=42,
    checkpoint_interval=10,
    checkpoint_dir='checkpoint',
    experiment_name='max_tgp'
)

hyperparameters = TGPHyperparameters(
    pop_size=10,
    max_size=MAX_SIZE,
    max_depth=MAX_DEPTH,
    cx_rate=0.9,
    mutation_rate=1.0/D,
    tournament_size=4,
    penalization_complexity_factor=0.0,
    erc=False
)

evals = []
for _ in range(NUM_INSTANCES):
    sleep(1)
    config.global_seed = int(time.time_ns())
    tgp = TinyTGP(functions, terminals, config, hyperparameters)
    best = tgp.evolve(problem)
    print(f"{tgp.expression(best.genome)}")
    evals.append(tgp.num_evaluations)

print("")
print(f"Mean: {np.mean(evals)}")
print(f"Median: {np.median(evals)}")

