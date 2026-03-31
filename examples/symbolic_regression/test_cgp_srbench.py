from src.benchmark.symbolic_regression.srbench import SRBench
import numpy as np
import random
from pmlb import fetch_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from src.gp.tinyverse import GPConfig
from src.gp.tiny_ge import TinyGE,  GEHyperparameters
from src.gp.functions import ADD, SUB, MUL, DIV, EXP, LOG, SQRT, SQR, CUBE
from src.gp.tiny_cgp import TinyCGP, CGPHyperparameters, CGPConfig 



MAXTIME = 3600  # 1 hour
MAXGEN = 100
POPSIZE = 100
datasets = ["192_vineyard"] 

functions=["+", "-", "*", "/", "exp", "log", "square", "cube"]

tree_ge_hyperparameters_baseline = CGPHyperparameters(
    mu=1,
    lmbda=32,
    population_size=33,
    levels_back=100,
    mutation_rate=0.1,
    strict_selection=True,
    cx_rate = 0.9,
    tournament_size = 7,
    num_function_nodes=10
)


tree_ge_hyperparameters_optimized = CGPHyperparameters(
    mu=1,
    lmbda=32,
    population_size=33,
    levels_back=100,
    mutation_rate=0.1,
    strict_selection=True,
    cx_rate = 0.9,
    tournament_size = 7,
    num_function_nodes=10
)

tree_ge_config = CGPConfig(
    num_jobs=1,
    max_generations=10,
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
    max_time=60,
    global_seed=42,
    checkpoint_interval=10,
    checkpoint_dir='examples/checkpoint',
    experiment_name='sr_cgp'
)

# tree_ge_config_opt = TreeGEConfig(
#     num_jobs=1,
#     max_generations=100,
#     stopping_criteria=1e-6,
#     minimizing_fitness=True,  # this should be used from the problem instance
#     ideal_fitness=1e-6,  # this should be used from the problem instance
#     silent_algorithm=False,
#     silent_evolver=False,
#     minimalistic_output=True,
#     num_outputs=1,
#     report_interval=1,
#     max_time=200,
#     global_seed=60,
#     checkpoint_interval=10,
#     checkpoint_dir='examples/checkpoint',
#     experiment_name='sr_3ge'
# )


# Generate a list of seeds, e.g., 5 runs
runs = 2
seeds = [random.randint(0, 10000) for _ in range(runs)]     # Generate a list of seeds, e.g., 5 runs

results = {}

for dataset in datasets:
    for run in range(runs):
        print(f"Running dataset: {dataset}\n")
        X, y = fetch_data(dataset, return_X_y=True)
        train_X, test_X, train_y, test_y = train_test_split(X, y, train_size=0.75)

        tree_ge_config.global_seed = seeds[run]

        tree_ge = SRBench(
            "CGP",
            tree_ge_config,
            tree_ge_hyperparameters_baseline,
            functions=functions
        )

        tree_ge_optimized = SRBench(
            "CGP",
            tree_ge_config,
            tree_ge_hyperparameters_optimized,
            functions=functions
        )

        # baseline 3ge model
        print("Baseline 3GE model")
        tree_ge.fit(train_X, train_y)
        expr = tree_ge.get_model()
        train_score = tree_ge.score(train_X, train_y)
        test_score = tree_ge.score(test_X, test_y)
        print(f"tree-ge expression: {expr}")
        print(f"tree_ge best fitness: {tree_ge.model.best_individual.fitness}")
        print(f"tree-ge train score: {train_score}")
        print(f"tree-ge test score: {test_score}")

        print("="*80)

        # optimized 3ge model
        print("\nOptimized 3GE model")
        tree_ge_optimized.fit(train_X, train_y)
        expr_opt = tree_ge_optimized.get_model()
        train_score_opt = tree_ge_optimized.score(train_X, train_y)
        test_score_opt = tree_ge_optimized.score(test_X, test_y)
        print(f"tree_ge optimized expression: {expr_opt}")
        print(f"tree_ge optimized best fitness: {tree_ge_optimized.model.best_individual.fitness}")
        print(f"tree_ge optimized train score: {train_score_opt}")
        print(f"tree_ge optimized test score: {test_score_opt}")

        results[dataset] = {
            "baseline": {
                "expression": expr,
                "train_score": train_score,
                "test_score": test_score
            },
            "optimized": {
                "expression": expr_opt,
                "train_score": train_score_opt,
                "test_score": test_score_opt
            }
        }