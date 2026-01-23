from time import sleep

import numpy as np
from matplotlib import pyplot as plt
from src.analysis.problems import MaxPlusMul, MaxPlus
from src.gp.tiny_cgp import *
from src.gp.functions import ADD, MUL
from src.gp.tiny_tgp import TGPConfig
from src.gp.tinyverse import Const
from src.analysis.models.simple_tgp import SimpleTGP, SimpleTGPHyperparameters

NUM_INSTANCES = 30
MAX_GENERATIONS = 5000000
D = 10
T = 1
MAX_SIZE = math.pow(2, D + 1)
MAX_DEPTH = D + 1
functions = [ADD]
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
    report_interval=100,
    max_time=3600,
    global_seed=None,
    checkpoint_interval=9999999,
    checkpoint_dir='../checkpoint',
    experiment_name='max_tgp'
)

hyperparameters = SimpleTGPHyperparameters(
    lmbda=1,
    k=1,
    strict_selection = True
)


x = []
y = []
for d in range(1,D+1):
    evals = []
    for _ in range(NUM_INSTANCES):
        problem = MaxPlus(d=d, t=T)
        config.ideal_fitness = problem.ideal
        config.global_seed = int(time.time_ns())
        tgp = SimpleTGP(functions, terminals, config, hyperparameters)
        best = tgp.evolve(problem)
        evals.append(tgp.num_evaluations)

    avg = np.mean(evals)
    x.append(d)
    y.append(avg)
    print(f"{d};{avg}")


fig1, ax1 = plt.subplots()
ax1.plot(x, y, linewidth=2.0)
ax1.set_yscale('linear')
ax1.set_xscale('linear')
plt.xlabel("D")
plt.ylabel("# Iterations")
plt.show()

fig2, ax2 = plt.subplots()
ax2.plot(x, y, linewidth=2.0)
ax2.set_yscale('log')
ax2.set_xscale('linear')
plt.xlabel("D")
plt.ylabel("# Iterations")
plt.show()
