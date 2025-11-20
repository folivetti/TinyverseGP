"""
Example module to test TinyverseGP on SRBench.

More information about SRBench can be obtained here:

https://cavalab.org/srbench/
https://github.com/cavalab/srbench/tree/master
"""

from backup.tiny_gp import Var
from src.benchmark.symbolic_regression.srbench import SRBench
import numpy as np
from pmlb import fetch_data
from sklearn.model_selection import train_test_split
from src.gp.tiny_cgp import CGPConfig, CGPHyperparameters, TinyCGP
from src.gp.tiny_tgp import TGPHyperparameters, TGPConfig, TinyTGP
from src.gp.tiny_ge import GEConfig, GEHyperparameters, TinyGE
from src.gp.tiny_lgp import LGPConfig, LGPHyperparameters, TinyLGP
import argparse
import csv
from src.gp.problem import Problem, BlackBox
from src.gp.loss import mean_squared_error, linear_scaling_mse
from src.gp.functions import ADD, SUB, MUL, DIV, EXP, LOG, SQR, CUBE
from src.hpo.hpo import SMAC4SRBenchInterface, SMACInterface
from src.gp.tinyverse import Const

parser = argparse.ArgumentParser(
                                prog='SRBench Test Scenario',
                                description='Run algorithm on selected SRBench instances',
                                epilog='')

parser.add_argument('--maxtime', dest='maxtime', type=int, default=3600)
parser.add_argument('--maxgen', dest='maxgen', type=int, default=100)
parser.add_argument('--popsize', dest='popsize', type=int, default=100)
parser.add_argument('--algo', dest='algo', type=str, default="TGP")
parser.add_argument('-d', '--dataset', dest='dataset', type=str, default="")
parser.add_argument('-s', '--seed', dest='seed', type=int, default=42)
parser.add_argument('-o', '--optimise', action='store_true')
parser.add_argument('-v', '--verbose', action='store_true')  # on/off flag
args = parser.parse_args()

if args.dataset != "":
    group_datasets = [[args.dataset]]
else:
    group_datasets = [['522_pm10', '678_visualizing_environmental', '192_vineyard', '1028_SWD'],
                  ['1199_BNG_echoMonths', '210_cloud', '1089_USCrime', '1193_BNG_lowbwt'],
                  ['557_analcatdata_apnea1', '650_fri_c0_500_50', '579_fri_c0_250_5', '606_fri_c2_1000_10']
                 ]

strfunctions = ['+','-','*','/','exp','log','square','cube']
functions = [ADD, SUB, MUL, DIV, EXP, LOG, SQR, CUBE] 
terminals=[1,0.5,np.pi, np.sqrt(2)]
print(args)
if args.algo=='TGP':
    # Set up hyperparameters for TGP and CGP 
    hyperparams = TGPHyperparameters(
        max_depth=8,
        max_size=100,
        pop_size=args.popsize,
        tournament_size=3,
        mutation_rate=0.2,
        cx_rate=0.9,
        erc=False # ephemeral random constants
    )
    
    #   Set up configurations for TGP and CGP 
    config = TGPConfig( num_jobs=1,
                            max_generations=args.maxgen,
                            stopping_criteria=1e-6,
                            minimizing_fitness=True, 
                            ideal_fitness=1e-16,
                            silent_algorithm=True,
                            silent_evolver=True,
                            minimalistic_output=True,
                            num_outputs=1,
                            report_interval=1000000,
                            max_time=args.maxtime,
                            global_seed=args.seed,
                            checkpoint_interval=10,
                            checkpoint_dir='checkpoints',
                            experiment_name='srbench_tgp'
                        )
    scaling = False
    
elif args.algo=='CGP':
    hyperparams = CGPHyperparameters(
        mu=2,
        lmbda=10,
        strict_selection=True,
        mutation_rate=0.3,
        pop_size=args.popsize,
        num_function_nodes=30,
        levels_back=10
    )


    config = CGPConfig(
                            num_jobs=1,
                            max_generations=args.maxgen,
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
                            report_interval=10,
                            max_time=args.maxtime,
                            global_seed=args.seed,
                            checkpoint_interval=10,
                            checkpoint_dir='checkpoints',
                            experiment_name='srbench_cgp'
                        )
    scaling = True
    
elif args.algo == "LGP":
    hyperparams = LGPHyperparameters(mu=2, lmbda=10, strict_selection=True, mutation_rate=0.3, pop_size=args.popsize, num_function_nodes=30, levels_back=10)

    config = LGPConfig(
        num_jobs=1,
        max_generations=args.maxgen,
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
        report_interval=10,
        max_time=args.maxtime,
        global_seed=args.seed,
        checkpoint_interval=10,
        checkpoint_dir="checkpoints",
        experiment_name="srbench_cgp",
    )
    scaling = True
    
elif args.algo == "GE":
    hyperparams = GEHyperparameters(mu=2, lmbda=10, strict_selection=True, mutation_rate=0.3, pop_size=args.popsize, num_function_nodes=30, levels_back=10)

    config = GEConfig(
        num_jobs=1,
        max_generations=args.maxgen,
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
        report_interval=10,
        max_time=args.maxtime,
        global_seed=args.seed,
        checkpoint_interval=10,
        checkpoint_dir="checkpoints",
        experiment_name="srbench_cgp",
    )
    scaling = True
#cgp_config.init()
else:
    print(f"{args.algo} is not a supported algorithm")
    exit()



    
for g in group_datasets:
    for d in g:
        csv_filename = f"experiments_scripts/output_data/{d}_{args.algo}_{args.seed}.csv"
        with open(csv_filename, mode="w", newline="") as csv_file:
            fieldnames = ["dataset_name", "algo_name", "nb_trials", "def_parameters", "opt_parameters", "def_train", "def_test", "opt_train", "opt_test", "seed"]
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            print(f"Running dataset: {d}\n")
            X, y = fetch_data(d, return_X_y=True)
            train_X, test_X, train_y, test_y = train_test_split(X, y, train_size=0.75, random_state=1337)
            
            # Default run
            algo = SRBench(args.algo, config, hyperparams, functions=strfunctions, terminals=terminals, scaling_=scaling)
            algo.fit(train_X, train_y)
            def_train_score = algo.score(train_X, train_y)
            def_test_score = algo.score(test_X, test_y)
            
            writer.writerow({
                'dataset_name': d,
                'algo_name': args.algo,
                'nb_trials': 0,
                'def_parameters': str(hyperparams.__dict__),
                'opt_parameters': str(hyperparams.__dict__),
                'def_train': def_train_score,
                'def_test': def_test_score,
                'opt_train': def_train_score,
                'opt_test': def_test_score,
                'seed': args.seed
            })

            algo = SRBench(args.algo, config, hyperparams, functions=strfunctions, terminals=terminals, scaling_=scaling)
            if args.optimise:
                

                ## Perform HPO via SMAC
                config.silent_algorithm=True
                config.silent_evolver=True
                trials = 200
                
                # try GPModel HPO (does not work for now)
                # interface = SMACInterface()
                # problem=BlackBox(train_X, train_y, algo.loss, 1e-16, True)
                # pb_terminals = [Var(i) for i in range(train_X.shape[1])] + [Const(c) for c in terminals]
                # loss = linear_scaling_mse if scaling else mean_squared_error
                # if args.algo=='CGP':
                #     model = TinyCGP(functions, pb_terminals, config, hyperparams)
                # elif args.algo=='TGP':   
                #     model = TinyTGP(functions, pb_terminals, config, hyperparams)
                # opt_hyperparameters = interface.optimise(model,problem, n_trials_=trials)
                
                # try SRBench HPO
                algo = SRBench(args.algo, config, hyperparams, functions=strfunctions, terminals=terminals, scaling_=scaling)
                interface_srbench = SMAC4SRBenchInterface(algo)
                opt_hyperparameters_srbench = interface_srbench.optimise(
                    train_X,
                    train_y,
                    n_trials=trials,
                    seed=args.seed,
                    dataset_name=d
                )
                
                algo = SRBench(args.algo, config, opt_hyperparameters_srbench, functions=strfunctions, terminals=terminals, scaling_=scaling)
                algo.fit(train_X, train_y) 
                opt_train_score = algo.score(train_X, train_y)
                opt_test_score = algo.score(test_X, test_y)
                
                writer.writerow(
                    {
                        "dataset_name": d,
                        "algo_name": args.algo,
                        "nb_trials": trials,
                        "def_parameters": str(hyperparams.__dict__),
                        "opt_parameters": str(opt_hyperparameters_srbench.__dict__),
                        "def_train": def_train_score,
                        "def_test": def_test_score,
                        "opt_train": opt_train_score,
                        "opt_test": opt_test_score,
                        "seed": args.seed,
                    }
                )


