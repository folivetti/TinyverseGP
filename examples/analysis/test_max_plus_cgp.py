import matplotlib.pyplot as plt
import numpy as np

from src.analysis.models.simple_cgp import SimpleCGP
from src.analysis.problems import MaxPlusMul, MaxPlus
from src.gp.tiny_cgp import *
from src.gp.functions import ADD, MUL
from src.gp.tinyverse import Const

NUM_INSTANCES = 30
D = 15
T = 1
problem = MaxPlus(d=D, t=T)
functions = [ADD]
terminals = [Const(T)]
ideal = problem.ideal

config = CGPConfig(
    num_jobs=1,
    max_generations=5000000,
    stopping_criteria=ideal,
    minimizing_fitness=False,
    ideal_fitness=ideal,
    silent_algorithm=True,
    silent_evolver=True,
    minimalistic_output=True,
    num_functions=len(functions),
    max_arity=2,
    num_inputs=1,
    num_outputs=1,
    report_interval=100000,
    max_time=3600,
    global_seed=42,
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
        problem = MaxPlus(d=d, t=T)
        #print(f"Maximum value: {problem.ideal}")
        config.ideal_fitness = problem.ideal
        config.global_seed = int(time.time_ns())
        cgp = SimpleCGP(functions, terminals, config, hyperparameters)
        best = cgp.evolve(problem)
        evals.append(cgp.num_evaluations)

    avg = np.mean(evals)
    x.append(d)
    y.append(avg)
    print(f"{d};{avg}")


fig, ax = plt.subplots()
ax.plot(x, y, linewidth=2.0)
ax.set_yscale('log')
ax.set_xscale('log')
plt.xlabel("D")
plt.ylabel("# Iterations")
plt.show()