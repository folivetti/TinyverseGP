"""
Script for performing HPO via SMAC3 for CGP on symbolic regression problems with subsequent evaluation.  
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
import sys
import copy

# Seed for HPO
smac_seed = 42

# Get input for seed, genetic operator, dataset
seed = int(sys.argv[1])
operator_ = str(sys.argv[2])
dataset = str(sys.argv[3])

print(f"This is job number {seed}")
print(f"Dataset: {dataset}")

# Set function set and terminal set
functions = ['+','-','*','/','exp','log','square','cube']
terminals=[1,0.5,np.pi, np.sqrt(2)]

X, y = fetch_data(dataset, return_X_y=True)

# Configuration for the CGP evolutionary run
cgp_config = CGPConfig(
    num_jobs=1,
    max_generations=200,
    max_generations=200,
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
    max_time=600,
    global_seed=smac_seed
    max_time=600,
    global_seed=smac_seed
)

# Hyperparameter settings for CGP initialisation
cgp_hyperparams = CGPHyperparameters(
    mu=1,
    lmbda=99,
    population_size=1,
    levels_back=100,
    lmbda=99,
    population_size=1,
    levels_back=100,
    mutation_rate=0.1,
    strict_selection=True,
    cx_rate = 0.7,
    tournament_size = 7,
    num_function_nodes = 10,
    operator = operator_
)

# Set the number of genes per node and the total number of genes
cgp_config.init() 

train_X, test_X, train_y, test_y = train_test_split(X, y, train_size=0.75, random_state=smac_seed)

# SMAC3 settings
n_trials = 200
n_splits = 5 # Number of splits for k-fold CV

# CGP initialisation
cgp = SRBench('CGP', cgp_config, cgp_hyperparams, functions=functions, terminals=terminals, scaling_=False)
cgp.fit(train_X, train_y)

# HPO
interface = SMACInterface()
opt_hyperparameters = interface.optimise(gpmodel_=cgp.model, dataset_=dataset, train_X_=train_X, train_y_=train_y, n_trials_=n_trials, seed_=seed, n_splits_=n_splits)

print("="*50)
print(f"Final incumbent: {opt_hyperparameters}")
print("="*50)

# CGP runs
n_runs = 30

# Test SMAC3 optimised configuration over n_runs runs
print("OPTIMISED HYPERPARAMETERS' PERFORMANCE")
print("="*50)

results_train = {
    "fitness": [],
    "r2_score": []
}

results_test = {
    "fitness": [],
    "r2_score": []
}

for i in range(n_runs):
    cgp_opt = SRBench(
            "CGP",
            cgp_config,
            opt_hyperparameters, # Use SMAC3-optimised configuration
            functions=functions,
            terminals=terminals,
            scaling_=False
        )
    
    cgp_opt.config.global_seed = i + 1 # Use different seeds for every run

    cgp_opt.fit(train_X, train_y) # Evolve on training set 

    # Assess train performance
    results_train["fitness"].append(cgp_opt.model.best_individual.fitness)
    results_train["r2_score"].append(cgp_opt.score(train_X, train_y))

    # Evaluate best individual on test set
    cgp_ = copy.deepcopy(cgp_opt.model)
    best_individual_genome = cgp_opt.model.best_individual.genome
    problem  = BlackBox(test_X, test_y, mean_squared_error, 1e-16, True)
    test_fitness = problem.evaluate(best_individual_genome, cgp_)

    # Assess test performance
    results_test["fitness"].append(test_fitness)
    results_test["r2_score"].append(cgp_opt.score(test_X, test_y))

print(f"Results (train set): {results_train}")
print(f"Results (test set): {results_test}")
print("="*50)

# Calculate fitness-related measures for the training set
mean_best_fitness = np.mean(results_train["fitness"])
best_fitness_var = np.var(results_train["fitness"])
best_fitness_std = np.std(results_train["fitness"])
best_fitness_quart = np.quantile(results_train["fitness"], [0.25, 0.5, 0.75])

print(f"Mean best fitness (old config, train_set): {mean_best_fitness}")
print(f"Variance of best fitness (old config, train_set): {best_fitness_var}")
print(f"Standard deviation of best fitness (old config, train_set): {best_fitness_std}")
print(f"Quartiles of best fitness (old config, train_set): {best_fitness_quart}")
print("="*50)

# Calculate fitness-related measures for the test set
mean_fitness = np.mean(results_test["fitness"])
fitness_var = np.var(results_test["fitness"])
fitness_std = np.std(results_test["fitness"])
fitness_quart = np.quantile(results_test["fitness"], [0.25, 0.5, 0.75])

print(f"Mean fitness (old config, test set): {mean_fitness}")
print(f"Variance of fitness (old config, test set): {fitness_var}")
print(f"Standard deviation of fitness (old config, test set): {fitness_std}")
print(f"Quartiles of fitness (old config, test set): {fitness_quart}")
print("="*50)

print(f"The following seed was used: {seed}")
print(f"The following number of trials was used: {n_trials}")
print(f"The following operator was used: {operator_}")
print(f"The following dataset was used: {dataset}")
print("="*50)