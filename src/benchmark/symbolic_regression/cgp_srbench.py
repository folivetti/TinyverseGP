from src.benchmark.symbolic_regression.srbench import SRBench
import numpy as np
from pmlb import fetch_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from src.gp.tiny_cgp import CGPConfig, CGPHyperparameters

group_datasets = [
    ["192_vineyard", "522_pm10", "678_visualizing_environmental", "1028_SWD"],
    ["1199_BNG_echoMonths", "210_cloud", "1089_USCrime", "1193_BNG_lowbwt"],
    [
        "557_analcatdata_apnea1",
        "650_fri_c0_500_50",
        "579_fri_c0_250_5",
        "606_fri_c2_1000_10",
    ],
]

functions = ["+", "-", "*", "/", "exp", "log", "square", "cube"]
terminals = [1, 0.5, np.pi, np.sqrt(2)]

#   Set up configuration and CGP
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
    population_size=412,
    levels_back=76,
    mutation_rate=0.1781129697338,
    strict_selection=True,
    cx_rate = 0.7487612692639,
    tournament_size = 2,
    num_function_nodes = 42
)

for g in group_datasets:
    for d in g:
        print(f"Running dataset: {d}\n")
        X, y = fetch_data(d, return_X_y=True)
        train_X, test_X, train_y, test_y = train_test_split(X, y, train_size=0.75, shuffle=False)
        cgp = SRBench(
            "CGP",
            cgp_config,
            cgp_hyperparams,
            functions=functions,
            terminals=terminals,
            scaling_=False
        )
        cgp.config.num_inputs = X.shape[1]
        cgp.fit(
            train_X, train_y
        ) 

        print(cgp.get_model())
        print(f"cgp train score: {cgp.score(train_X, train_y)}")
        print(f"cgp test score: {cgp.score(test_X, test_y)}")
        print("=" * 50)

