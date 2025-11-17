"""
Example module to test TinyverseGP on SRBench.

More information about SRBench can be obtained here:

https://cavalab.org/srbench/
https://github.com/cavalab/srbench/tree/master
"""

from src.benchmark.symbolic_regression.srbench import SRBench
import numpy as np
from pmlb import fetch_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from src.gp.tinyverse import GPConfig
from src.gp.tiny_ge import TinyGE,  GEHyperparameters
from src.gp.functions import ADD, SUB, MUL, DIV, EXP, LOG, SQRT, SQR, CUBE
from src.gp.tiny_3ge import Tiny3GE, TreeGEHyperparameters, TreeGEConfig 
from src.gp.tiny_cgp import CGPConfig, CGPHyperparameters
from src.gp.tiny_tgp import TGPHyperparameters, TGPConfig
import ioh
from multiprocessing import Pool
from itertools import product

DATA_FOLDER = "."
MAXTIME = 3600  # 1 hour
MAXGEN = 100
POPSIZE = 100
group_datasets = [
    [ "192_vineyard", "1028_SWD"],  # "522_pm10, "678_visualizing_environmental""
    ["1199_BNG_echoMonths", "210_cloud", "1089_USCrime", "1193_BNG_lowbwt"],
    [
        "557_analcatdata_apnea1",
        "650_fri_c0_500_50",
        "579_fri_c0_250_5",
        "606_fri_c2_1000_10",
    ],
]

functions = [ADD, SUB, MUL, DIV, EXP, LOG, SQR, CUBE]
terminals = [1, 0.5, np.pi, np.sqrt(2)]

grammar = {
    "<expr>": [
        "ADD(<expr>, <expr>)",
        "SUB(<expr>, <expr>)",
        "MUL(<expr>, <expr>)",
        "DIV(<expr>, <expr>)",
        "EXP(<expr>)",
        "LOG(<expr>)",
        "SQR(<expr>)",
        # "CUBE(<expr>)",
        "<const>",
        "<var>",
    ],
    "<const>": ["1", "0.5", "3.14159", "1.41421"],  # ~pi, ~sqrt(2)
    "<var>": ["x"],
}

# Set up hyperparameters for TGP and CGP
tgp_hyperparams = TGPHyperparameters(
    max_depth=8,
    max_size=100,
    pop_size=POPSIZE,
    tournament_size=3,
    mutation_rate=0.2,
    cx_rate=0.9,
    erc=False,  # ephemeral random constants
)
cgp_hyperparams = CGPHyperparameters(
    mu=2,
    lmbda=10,
    num_function_nodes=10,
    population_size=POPSIZE,
    levels_back=10,
    strict_selection=True,
    mutation_rate=0.3,
)

treege_hyperparams = TreeGEHyperparameters(
    pop_size=100,
    min_depth=2,
    max_depth=6,
    codon_size=256,
    cx_rate=0.9,
    mutation_rate=0.1,
    tournament_size=2,
    penalty_value=99999,
)

ge_hyperparams = GEHyperparameters(
    pop_size=100,
    genome_length=40,
    codon_size=1000,
    cx_rate=0.9,
    mutation_rate=0.1,
    tournament_size=2,
    penalty_value=99999,
)


#   Set up configurations for TGP and CGP
tgp_config = TGPConfig(
    num_jobs=1,
    max_generations=MAXGEN,
    stopping_criteria=1e-6,
    minimizing_fitness=True,
    ideal_fitness=1e-16,
    silent_algorithm=True,
    silent_evolver=True,
    minimalistic_output=True,
    num_outputs=1,
    report_interval=1000000,
    max_time=MAXTIME,
    global_seed=42,
    checkpoint_interval=10,
    checkpoint_dir="examples/checkpoint",
    experiment_name="srbench_tgp",
)
cgp_config = CGPConfig(
    num_jobs=1,
    max_generations=MAXGEN,
    stopping_criteria=1e-16,
    minimizing_fitness=True,
    ideal_fitness=1e-16,
    silent_algorithm=True,
    silent_evolver=True,
    minimalistic_output=True,
    num_functions=len(functions),
    max_arity=2,
    num_inputs=1,
    num_outputs=1,
    # num_function_nodes=30,
    report_interval=10,
    max_time=MAXTIME,
    global_seed=42,
    checkpoint_interval=10,
    checkpoint_dir="examples/checkpoint",
    experiment_name="srbench_cgp",
)
# cgp_config.init()

treege_config = TreeGEConfig(
    num_jobs=1,
    max_generations=100,
    stopping_criteria=1e-6,
    minimizing_fitness=True,  # this should be used from the problem instance
    ideal_fitness=1e-6,  # this should be used from the problem instance
    silent_algorithm=False,
    silent_evolver=False,
    minimalistic_output=True,
    num_outputs=1,
    report_interval=1,
    max_time=200,
    global_seed=42,
    checkpoint_interval=10,
    checkpoint_dir='examples/checkpoint',
    experiment_name='sr_3ge'
)

ge_config = GPConfig(
    num_jobs=1,
    max_generations=100,
    stopping_criteria=1e-6,
    minimizing_fitness=True,  # this should be used from the problem instance
    ideal_fitness=1e-6,  # this should be used from the problem instance
    silent_algorithm=False,
    silent_evolver=False,
    minimalistic_output=True,
    num_outputs=1,
    report_interval=1,
    max_time=60,
    global_seed=42,
    checkpoint_interval=10,
    checkpoint_dir='examples/checkpoint',
    experiment_name='sr_ge'
)

def get_train_test_split(dataset_name, train_size=0.75, random_seed=42):
    """
    Get consistent train-test split for a dataset using fixed seed.
    
    Args:
        dataset_name: Name of the dataset from PMLB
        train_size: Proportion of data for training (default: 0.75)
        random_seed: Random seed for reproducibility (default: 42)
    
    Returns:
        train_X, test_X, train_y, test_y
    """
    X, y = fetch_data(dataset_name, return_X_y=True)
    
    # Shuffle data with fixed seed to ensure consistency
    np.random.seed(random_seed)
    indices = np.random.permutation(len(X))
    X_shuffled = X[indices]
    y_shuffled = y[indices]
    
    # Split without additional shuffling
    train_X, test_X, train_y, test_y = train_test_split(
        X_shuffled, y_shuffled, train_size=train_size, shuffle=False
    )
    
    return train_X, test_X, train_y, test_y


# Create dictionaries for configs, hyperparameters, and grammars
configs = {
    "TGP": tgp_config,
    "CGP": cgp_config,
    "3GE": treege_config,
    "GE": ge_config,
}

hyperparams = {
    "TGP": tgp_hyperparams,
    "CGP": cgp_hyperparams,
    "3GE": treege_hyperparams,
    "GE": ge_hyperparams,
}

all_datasets = [dataset for group in group_datasets for dataset in group]
all_datasets = {idx : name for idx, name in enumerate(all_datasets)}

def run_benchmark(arg):
    """
    Run a single method on a single dataset.
    
    Args:
        arg = tuple of method_name: One of ['TGP', 'CGP', '3GE', 'GE'], fid: Function ID and iid: instance id (train-test seed)
    """
    method_name, fid, iid = arg
    # Validate method name
    if method_name not in configs:
        raise ValueError(f"Unknown method: {method_name}. Choose from {list(configs.keys())}")
    dataset_name = all_datasets[fid]
    print(f"Running {method_name} on dataset: {dataset_name}\n")
    
    # Get consistent train-test split
    train_X, test_X, train_y, test_y = get_train_test_split(dataset_name, random_seed=iid)
    
    # Create model using dictionaries
    model_kwargs = {
        'config': configs[method_name],
        'hyperparameters': hyperparams[method_name],
        'functions': functions,
        'terminals': terminals,
        'scaling_': False,
    }
    
    # Only add grammar parameter for 3GE
    if method_name == '3GE':
        model_kwargs['grammar'] = grammar
    
    model = SRBench(method_name, **model_kwargs)
    
    # Create logger
    logger = ioh.logger.Analyzer(
        [ioh.logger.trigger.ALWAYS], 
        root=DATA_FOLDER, 
        folder_name=f"srbench_{method_name}", 
        algorithm_name=method_name
    )
    for _ in range(5):
    # Fit the model
        model.fit(train_X, train_y, logger=logger, fid=fid, name=f"{dataset_name}_S{iid}")
        model.hyperparameters['global_seed'] += 1
    
    # # Print results
    # print(model.get_model())
    # print(f"{method_name} train score: {model.score(train_X, train_y)}")
    # print(f"{method_name} test score: {model.score(test_X, test_y)}")
    # print("=" * 50)
    
    # return model

def runParallelFunction(runFunction, arguments):
    """
        Return the output of runFunction for each set of arguments,
        making use of as much parallelization as possible on this system

        :param runFunction: The function that can be executed in parallel
        :param arguments:   List of tuples, where each tuple are the arguments
                            to pass to the function
        :return:
    """
    

    arguments = list(arguments)
    p = Pool(min(128, len(arguments)))
    results = p.map(runFunction, arguments)
    p.close()
    return results


def run_all_benchmarks():
    """Run all methods on all datasets."""
    for fid in range(len(all_datasets)):
        # Run all methods on this dataset
        for method in ['CGP', 'TGP', 'GE', '3GE']:
            try:
                arg = (method, fid, 1)
                run_benchmark(arg)
            except Exception as e:
                print(f"Error running {method} on {fid}: {e}")

def run_all_parallel():
    fids = np.arange(len(all_datasets))
    methods = ['CGP', 'TGP', 'GE', '3GE']
    iids = np.arange(5)
    args = product(methods, fids, iids)
    runParallelFunction(run_benchmark, args)

if __name__ == "__main__":
    # To run all benchmarks:
    # run_all_benchmarks()
    run_all_parallel()

