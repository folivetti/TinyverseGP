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
import sys
import copy

seed = 42
operator_ = str(sys.argv[1])

dataset = "597_fri_c2_500_5"

functions = ['+','-','*','/','exp','log','square','cube']
terminals=[1,0.5,np.pi, np.sqrt(2)]

X, y = fetch_data(dataset, return_X_y=True)

cgp_config = CGPConfig(
    num_jobs=1,
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
    global_seed=seed
)


cgp_hyperparams = CGPHyperparameters(
    mu=1,
    lmbda=99,
    population_size=50,
    levels_back=200,
    mutation_rate=0.1,
    strict_selection=True,
    cx_rate = 0.7,
    tournament_size = 4,
    num_function_nodes = 200,
    operator = operator_
)

cgp_config.init() # Set the number of genes per node and the total number of genes

train_X, test_X, train_y, test_y = train_test_split(X, y, train_size=0.75, random_state=seed) # Pass seed for reproducibility

# CGP runs

n_runs = 30

# Test (old) baseline configuration over n_runs runs
print("BASELINE PERFORMANCE")
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
    cgp_old = SRBench(
            "CGP",
            cgp_config,
            cgp_hyperparams, # Use baseline configuration
            functions=functions,
            terminals=terminals,
            scaling_=False
        )
    
    cgp_old.config.global_seed = i + 1 # Use different seeds for every run

    cgp_old.fit(train_X, train_y) # Evolve on train set 

    # Assess train performance
    results_train["fitness"].append(cgp_old.model.best_individual.fitness)
    results_train["r2_score"].append(cgp_old.score(train_X, train_y))

    # Evaluate best individual on test set
    cgp_ = copy.deepcopy(cgp_old.model)
    best_individual_genome = cgp_old.model.best_individual.genome
    problem  = BlackBox(test_X, test_y, mean_squared_error, 1e-16, True)
    test_fitness = problem.evaluate(best_individual_genome, cgp_)

    # Assess test performance
    results_test["fitness"].append(test_fitness)
    results_test["r2_score"].append(cgp_old.score(test_X, test_y))

print(f"Results (train set): {results_train}")
print(f"Results (test set): {results_test}")
print("="*50)

# Calculate fitness-related measures for the train set
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

# Calculate R^2-score-related measures for the train set
mean_r2_score = np.mean(results_train["r2_score"])
r2_score_var = np.var(results_train["r2_score"])
r2_score_std = np.std(results_train["r2_score"])
r2_score_quart = np.quantile(results_train["r2_score"], [0.25, 0.5, 0.75])

print(f"Mean R2 score (old config, train_set): {mean_r2_score}")
print(f"Variance of R2 score (old config, train_set): {r2_score_var}")
print(f"Standard deviation of R2 score (old config, train_set): {r2_score_std}")
print(f"Quartiles of R2 score (old config, train_set): {r2_score_quart}")
print("="*50)

# Calculate R^2-score-related measures for the test set
mean_r2_score_test = np.mean(results_test["r2_score"])
r2_score_var_test = np.var(results_test["r2_score"])
r2_score_std_test = np.std(results_test["r2_score"])
r2_score_quart_test = np.quantile(results_test["r2_score"], [0.25, 0.5, 0.75])

print(f"Mean R2 score (old config, test set): {mean_r2_score_test}")
print(f"Variance of R2 score (old config, test set): {r2_score_var_test}")
print(f"Standard deviation of R2 score (old config, test set): {r2_score_std_test}")
print(f"Quartiles of R2 score (old config, test set): {r2_score_quart_test}")
print("="*50)

print(f"This is base config 1")
print(f"The following operator was used: {operator_}")
print("="*50)