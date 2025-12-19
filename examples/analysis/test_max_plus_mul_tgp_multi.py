import numpy as np
import csv
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
from src.analysis.problems import MaxPlusMul, MaxPlus
from src.gp.tiny_cgp import *
from src.gp.functions import ADD, MUL
from src.gp.tiny_tgp import TGPConfig
from src.gp.tinyverse import Const
from src.analysis.models.simple_tgp import SimpleTGP, SGPHyperparameters

NUM_INSTANCES = 30
MAX_GENERATIONS = 1000000
MAX_TIME = 9999999
EXPORT_CSV = True
PLOT = True
D_MIN = 1
D_MAX = 4
T = 1
MAX_SIZE = math.pow(2, D_MAX + 1)
MAX_DEPTH = D_MAX + 1
functions = [ADD, MUL]
terminals = [Const(T)]

sns.set_theme()

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
    max_time=MAX_TIME,
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


x = []
y = []
csv_data = []
for d in range(D_MIN, D_MAX + 1):
    evals = []
    deltas = []
    problem = MaxPlusMul(d=d, t=T)
    for _ in range(NUM_INSTANCES):
        config.ideal_fitness = problem.ideal
        config.global_seed = int(time.time_ns())
        tgp = SimpleTGP(functions, terminals, config, hyperparameters)
        t0 = time.time()
        best = tgp.evolve(problem)
        t1 = time.time()
        delta = t1 - t0
        evals.append(tgp.num_evaluations)
        deltas.append(delta)
        csv_data.append({'d': d, 'num_evals': tgp.num_evaluations})


    avg_eval = np.mean(evals)
    std = np.std(evals)
    avg_delta = np.mean(deltas)
    x.append(d)
    y.append(avg_eval)
    print(f"{d};{avg_eval:.2f};{std:.2f};{avg_delta:.2f}")



if EXPORT_CSV:
    with open('max_plus_mul_tgp.csv', 'w', newline='') as csvfile:
        fieldnames = ['d', 'num_evals']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_data)

if PLOT:
    data = pd.read_csv('max_plus_mul_tgp.csv')

    p = sns.lineplot(
        data=data,
        x="d", y="num_evals",
        markers=True,
    )

    p.set(xlabel='D', ylabel='# Iterations')
    p.set_xticks(range(D_MIN,D_MAX+1))
    p.set_xticklabels([str(d) for d in range(D_MIN,D_MAX+1)])
    plt.show()

