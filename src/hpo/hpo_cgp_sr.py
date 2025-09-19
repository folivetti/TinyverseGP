"""
Script to perform HPO via SMAC for CGP on a symbolic regression problem (557_analcatdata_apnea1)
"""
from sklearn.model_selection import train_test_split
import numpy as np
from pmlb import fetch_data
from src.gp.tiny_cgp import *
from src.gp.problem import BlackBox
from src.benchmark.symbolic_regression.sr_benchmark import SRBenchmark
from src.benchmark.symbolic_regression.srbench import SRBench
from src.gp.functions import *
from src.gp.loss import *
from src.gp.tinyverse import Var, Const
from src.hpo.hpo import SMACInterface

dataset = "192_vineyard"

functions = ['+','-','*','/','exp','log','square','cube']
terminals=[1,0.5,np.pi, np.sqrt(2)]

X, y = fetch_data(dataset, return_X_y=True)

cgp_config = CGPConfig(
    num_jobs=1,
    max_generations=100,
    stopping_criteria=1e-6,
    minimizing_fitness=True,
    ideal_fitness=1e-6,
    silent_algorithm=True,
    silent_evolver=True,
    minimalistic_output=True,
    num_functions=len(functions),
    max_arity=2,
    num_inputs=X.shape[1],
    num_outputs=1,
    report_interval=1,
    max_time=60
)


cgp_hyperparams = CGPHyperparameters(
    mu=1,
    lmbda=32,
    population_size=33,
    levels_back=len(terminals),
    mutation_rate=0.1,
    strict_selection=True,
    cx_rate = 0.9,
    tournament_size = 4,
    num_function_nodes = 10
)

cgp_config.init() # Set the number of genes per node and the total number of genes

loss = absolute_distance
train_X, test_X, train_y, test_y = train_test_split(X, y, train_size=0.75)
n_trials = 50

cgp = SRBench('CGP', cgp_config, cgp_hyperparams, functions=functions, terminals=terminals, scaling_=False)
cgp.fit(train_X, train_y)
interface = SMACInterface()

opt_hyperparameters = interface.optimise(cgp.model, n_trials)
print(opt_hyperparameters)

cgp_old = SRBench('CGP', cgp_config, cgp_hyperparams, functions=functions, terminals=terminals, scaling_=False)
cgp_old.fit(test_X, test_y)
print(cgp_old.model.expression(cgp_old.model.best_individual.genome))
print(f"old cgp train score: {cgp_old.score(train_X, train_y)}")
print(f"old cgp test score: {cgp_old.score(test_X, test_y)}")
print("="*50)

cgp = SRBench('CGP', cgp_config, opt_hyperparameters, functions=functions, terminals=terminals, scaling_=False)
cgp.fit(test_X, test_y)
print(cgp.model.expression(cgp.model.best_individual.genome))
print(f"cgp train score: {cgp.score(train_X, train_y)}")
print(f"cgp test score: {cgp.score(test_X, test_y)}")
print("="*50)