from src.benchmark.symbolic_regression.srbench import SRBench
import numpy as np
from pmlb import fetch_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from src.gp.tiny_cgp import CGPConfig, CGPHyperparameters

dataset = "579_fri_c0_250_5"

functions = ["+", "-", "*", "/", "exp", "log", "square", "cube"]
terminals = [1, 0.5, np.pi, np.sqrt(2)]

n_runs = 30

results = {
    "best_fitness": [],
    "train_score": [],
    "test_score": []
}

# Set up configuration and CGP
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
    num_inputs=1,
    num_outputs=1,
    report_interval=1,
    max_time=600,
    global_seed=42
)

cgp_config.init() # Set genes_per_node

# Set up hyperparameters for TGP and CGP
cgp_hyperparams = CGPHyperparameters(
    mu=1,
    lmbda=99,
    population_size=417,
    levels_back=1,
    mutation_rate=0.6792287230492,
    strict_selection=True,
    cx_rate = 0.4969017580152,
    tournament_size = 9,
    num_function_nodes = 31
)

X, y = fetch_data(dataset, return_X_y=True)
train_X, test_X, train_y, test_y = train_test_split(X, y, train_size=0.75, shuffle=False)
for i in range(n_runs):
    cgp = SRBench(
            "CGP",
            cgp_config,
            cgp_hyperparams,
            functions=functions,
            terminals=terminals,
            scaling_=False
        )
    
    cgp.config.num_inputs = X.shape[1]
    cgp.config.global_seed = i + 1

    cgp.fit(train_X, train_y) 

    results["best_fitness"].append(cgp.model.best_individual.fitness)
    results["train_score"].append(cgp.score(train_X, train_y))
    results["test_score"].append(cgp.score(test_X, test_y))

print(results)

mean__best_fitness = np.mean(results["best_fitness"])
mean_train_score = np.mean(results["train_score"])
mean_test_score = np.mean(results["train_score"])

best_fitness_var = np.var(results["best_fitness"])
train_score_var = np.var(results["train_score"])
test_score_var = np.var(results["test_score"])

best_fitness_quart = np.quantile(results["best_fitness"], [0.25, 0.5, 0.75])
train_score_quart = np.quantile(results["train_score"], [0.25, 0.5, 0.75])
test_score_quart = np.quantile(results["test_score"], [0.25, 0.5, 0.75])

print("="*50)
print(f"Mean best fitness: {mean__best_fitness}")
print(f"Mean train score: {mean_train_score}")
print(f"Mean test score: {mean_test_score}")
print("="*50)
print(f"Best fitness variance: {best_fitness_var}")
print(f"Train score variance: {train_score_var}")
print(f"Test score variance: {test_score_var}")
print("="*50)
print(f"Best fitness quartiles: {best_fitness_quart}")
print(f"Train score quartiles: {train_score_quart}")
print(f"Test score quartiles: {test_score_quart}")
print("="*50)
print(f"This is SMAC3 configuration 7 - Operator: {cgp.hyperparameters.operator}.")