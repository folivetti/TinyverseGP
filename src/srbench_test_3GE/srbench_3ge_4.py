from src.benchmark.symbolic_regression.srbench import SRBench
import numpy as np
import random
from pmlb import fetch_data
from sklearn.model_selection import train_test_split
from src.gp.functions import ADD, SUB, MUL, DIV, EXP, LOG, SQR, CUBE
from src.gp.tiny_3GE import TreeGEHyperparameters, TreeGEConfig
import statistics
import pprint

# Configuration
datasets = ["656_fri_c1_100_5"]
runs = 32
functions = [ADD, SUB, MUL, DIV, EXP, LOG, SQR, CUBE]

grammar = {
    "<expr>": [
        "ADD(<expr>, <expr>)",
        "SUB(<expr>, <expr>)",
        "MUL(<expr>, <expr>)",
        "DIV(<expr>, <expr>)",
        "EXP(<expr>)",
        "LOG(<expr>)",
        "SQR(<expr>)",
        "CUBE(<expr>)",
        "<const>",
        "<var>",
    ],
    "<const>": ["1", "0.5", "3.14159", "1.41421", "2", "3", "4", "5", "6", "7", "8", "9"],
    "<var>": [],
}

tree_ge_hyperparameters_baseline = TreeGEHyperparameters(
    pop_size=256, 
    min_depth=2, 
    max_depth=8, 
    codon_size=256,
    cx_rate=0.9, 
    mutation_rate=0.1, 
    tournament_size=2, 
    penalty_value=99999
)

tree_ge_hyperparameters_optimized = TreeGEHyperparameters(
    pop_size=458,
    mutation_rate=0.203, 
    cx_rate=0.968,
    tournament_size=9,
    min_depth=2,
    max_depth=8,
    rhh_rate=0.588,
    codon_size=256,
    penalty_value=99999
)

tree_ge_config = TreeGEConfig(
    num_jobs=1, 
    max_generations=100, 
    stopping_criteria=1e-6,
    minimizing_fitness=True, 
    ideal_fitness=1e-6,
    silent_algorithm=False, 
    silent_evolver=False,
    minimalistic_output=True, 
    num_outputs=1,
    report_interval=1, 
    max_time=16000,
    global_seed=42, 
    checkpoint_interval=10,
    checkpoint_dir='examples/checkpoint', 
    experiment_name='sr_3ge'
)

# generate seeds for repeated runs
seeds = [random.randint(0, 10000) for _ in range(runs)]

results = {}

# ---------------------------------------------------------------------
# Run experiments
# ---------------------------------------------------------------------
for dataset in datasets:
    print(f"\n=== Running dataset: {dataset} ===\n")
    X, y = fetch_data(dataset, return_X_y=True)
    train_X, test_X, train_y, test_y = train_test_split(X, y, train_size=0.75)

    results[dataset] = {"baseline": [], "optimized": []}

    for run_idx, seed in enumerate(seeds, start=1):
        print(f"--- Run {run_idx} (Seed={seed}) ---")
        tree_ge_config.global_seed = seed

        # Baseline model
        model_base = SRBench("3GE", tree_ge_config, tree_ge_hyperparameters_baseline,
                             functions=functions, grammar=grammar)
        model_base.fit(train_X, train_y)
        train_score_base = model_base.score(train_X, train_y)
        test_score_base = model_base.score(test_X, test_y)
        expr_base = model_base.get_model()
        best_fitness_base = model_base.model.best_individual.fitness
        results[dataset]["baseline"].append({
            "seed": seed, 
            "expression": expr_base,
            "train": train_score_base, 
            "test": test_score_base,
            "fitness": best_fitness_base
        })

        print(f"Baseline Expression: {expr_base}")
        print(f"Best Fitness: {best_fitness_base:.6f}")
        print(f"Train Score: {train_score_base:.6f} | Test Score: {test_score_base:.6f}")
        print("-"*60)

        # Optimized model
        model_opt = SRBench("3GE", tree_ge_config, tree_ge_hyperparameters_optimized,
                            functions=functions, grammar=grammar)
        model_opt.fit(train_X, train_y)
        train_score_opt = model_opt.score(train_X, train_y)
        test_score_opt = model_opt.score(test_X, test_y)
        expr_opt = model_opt.get_model()
        best_fitness_opt = model_opt.model.best_individual.fitness
        results[dataset]["optimized"].append({
            "seed": seed, 
            "expression": expr_opt,
            "train": train_score_opt, 
            "test": test_score_opt,
            "fitness": best_fitness_opt
        })

        print(f"Optimized Expression: {expr_opt}")
        print(f"Best Fitness: {best_fitness_opt:.6f}")
        print(f"Train Score: {train_score_opt:.6f} | Test Score: {test_score_opt:.6f}")
        print("="*80)

# ---------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------
def summarize(scores):
    mean_val = statistics.mean(scores)
    std_val = statistics.stdev(scores) if len(scores) > 1 else 0.0
    return f"{mean_val:.6f} +/- {std_val:.6f}"

print("\n=== Full results dictionary ===\n")
pprint.pprint(results)

for dataset in datasets:
    print(f"\n=== Summary for dataset: {dataset} ===")
    for model_label in ["baseline", "optimized"]:
        train_scores = [r["train"] for r in results[dataset][model_label]]
        test_scores = [r["test"] for r in results[dataset][model_label]]
        fitness_scores = [r["fitness"] for r in results[dataset][model_label]]
        print(f"\n{model_label.capitalize()}:")
        print(f"Train Score: {summarize(train_scores)}")
        print(f"Test Score:  {summarize(test_scores)}")
        print(f"Fitness:     {summarize(fitness_scores)}")

    print("\n--- Best expressions per model (lowest test score) ---")
    for model_label in ["baseline", "optimized"]:
        best_run = min(results[dataset][model_label], key=lambda r: r["fitness"])
        print(f"{model_label.capitalize()}: {best_run['expression']} | Fitness={best_run['fitness']:.6f}")

# ---------------------------------------------------------------------
# Print full dictionary of all results
# ---------------------------------------------------------------------
# print("\n=== Full results dictionary ===\n")
# pprint.pprint(results)
