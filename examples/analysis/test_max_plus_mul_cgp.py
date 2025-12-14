import numpy as np
from matplotlib import pyplot as plt

from src.analysis.models.simple_cgp import SimpleCGP
from src.analysis.problems import MaxPlusMul
from src.gp.tiny_cgp import *
from src.gp.functions import ADD, MUL
from src.gp.tinyverse import Const

NUM_INSTANCES = 100
MAX_GENERATIONS = 5000000
D = 30
T = 1
problem = MaxPlusMul(d=D, t=T)
functions = [ADD, MUL]
terminals = [Const(T), Const(0)]
ideal = problem.ideal


config = CGPConfig(
    num_jobs=100,
    max_generations=5000000,
    stopping_criteria=ideal,
    minimizing_fitness=False,
    ideal_fitness=ideal,
    silent_algorithm=True,
    silent_evolver=True,
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

x = []
y = []
for d in range(3,D+1):
    evals = []
    for _ in range(NUM_INSTANCES):
        problem = MaxPlusMul(d=d, t=T)
        config.ideal_fitness = problem.ideal
        config.global_seed = int(time.time_ns())
        cgp = SimpleCGP(functions, terminals, config, hyperparameters)
        best = cgp.evolve(problem)
        evals.append(cgp.num_evaluations)

    avg = np.mean(evals)
    std = np.std(evals)
    x.append(d)
    y.append(avg)
    print(f"{d};{avg};{std}")


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