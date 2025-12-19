import numpy as np
import csv
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt

from src.analysis.models.simple_cgp import SimpleCGP
from src.analysis.problems import MaxPlusMul, MaxPlus
from src.gp.tiny_cgp import *
from src.gp.functions import ADD, MUL
from src.gp.tiny_tgp import TGPConfig
from src.gp.tinyverse import Const
from src.analysis.models.simple_tgp import SimpleTGP, SGPHyperparameters

NUM_INSTANCES = 30
MAX_GENERATIONS = 5000000
EXPORT_CSV = True
PLOT = True
D_MIN = 1
D_MAX = 4
T = 1
MAX_SIZE = math.pow(2, D_MAX + 1)
MAX_DEPTH = D_MAX + 1
functions = [ADD, MUL]
terminals_tgp = [Const(T)]
terminals_cgp = [Const(T), Const(0)]

sns.set_theme()

config_tgp = TGPConfig(
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

config_cgp = CGPConfig(
    num_jobs=1,
    max_generations=5000000,
    stopping_criteria=None,
    minimizing_fitness=False,
    ideal_fitness=None,
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

hp_cgp = CGPHyperparameters(
    mu=1,
    lmbda=1,
    population_size=2,
    num_function_nodes=D_MAX,
    levels_back=D_MAX,
    mutation_rate=None,
    strict_selection=False,
)

hp_tgp = SGPHyperparameters(
    lmbda=1,
    k=1,
    strict_selection = False
)

x = []
y = []
csv_data = []
for d in range(D_MIN, D_MAX + 1):
    problem = MaxPlusMul(d=d, t=T)
    evals_cgp = []
    evals_tgp = []
    for _ in range(NUM_INSTANCES):
        config_tgp.ideal_fitness = problem.ideal
        config_tgp.global_seed = int(time.time_ns())

        config_cgp.ideal_fitness = problem.ideal
        config_cgp.global_seed = int(time.time_ns())

        tgp = SimpleTGP(functions, terminals_tgp, config_tgp, hp_tgp)
        cgp = SimpleCGP(functions, terminals_cgp, config_cgp, hp_cgp)

        tgp.evolve(problem)
        cgp.evolve(problem)

        csv_data.append({'d': d, 'model': 'simple_tgp', 'num_evals': tgp.num_evaluations})
        csv_data.append({'d': d, 'model': 'simple_cgp', 'num_evals': cgp.num_evaluations})

        evals_cgp.append(cgp.num_evaluations)
        evals_tgp.append(tgp.num_evaluations)

    avg_eval_cgp = np.mean(evals_cgp)
    avg_eval_tgp = np.mean(evals_tgp)
    print(f"{d};{avg_eval_cgp:.2f};simple_cgp")
    print(f"{d};{avg_eval_tgp:.2f};simple_tgp")
    print("")

if EXPORT_CSV:
    with open('max_plus_mul_tgp_cgp.csv', 'w', newline='') as csvfile:
        fieldnames = ['d', 'model', 'num_evals']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_data)

if PLOT:
    data = pd.read_csv('max_plus_mul_tgp_cgp.csv')

    p = sns.lineplot(
        data=data,
        x="d", y="num_evals",
        hue="model", style="model",
        markers=True,
    )

    p.set(xlabel='D', ylabel='# Iterations')
    p.set_xticks(range(D_MIN,D_MAX+1))
    p.set_xticklabels([str(d) for d in range(D_MIN,D_MAX+1)])
    plt.show()