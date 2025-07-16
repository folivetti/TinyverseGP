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
from src.gp.tiny_cgp import CGPConfig, CGPHyperparameters, TinyCGP
from src.gp.tiny_tgp import TGPHyperparameters, TGPConfig, TinyTGP
import argparse
from src.gp.problem import Problem, BlackBox

from src.hpo.hpo import SMACInterface


parser = argparse.ArgumentParser(
                                prog='SRBench Test Scenario',
                                description='Run TGP or CGP on selected SRBench instances',
                                epilog='')

parser.add_argument('--maxtime', dest='maxtime', type=int, default=3600)
parser.add_argument('--maxgen', dest='maxgen', type=int, default=100)
parser.add_argument('--popsize', dest='popsize', type=int, default=100)
parser.add_argument('--algo', dest='algo', type=str, default="TGP")
parser.add_argument('-s', '--seed', dest='seed', type=int, default=42)
parser.add_argument('-o', '--optimise', action='store_true')
parser.add_argument('-v', '--verbose',
                    action='store_true')  # on/off flag


args = parser.parse_args()

group_datasets = [['522_pm10', '678_visualizing_environmental', '192_vineyard', '1028_SWD'],
                  ['1199_BNG_echoMonths', '210_cloud', '1089_USCrime', '1193_BNG_lowbwt'],
                  ['557_analcatdata_apnea1', '650_fri_c0_500_50', '579_fri_c0_250_5', '606_fri_c2_1000_10']
                 ]

functions = ['+','-','*','/','exp','log','square','cube']
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
                            global_seed=42,
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
        population_size=args.popsize,
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
                            num_function_nodes=30,
                            report_interval=10,
                            max_time=args.maxtime,
                            global_seed=42,
                            checkpoint_interval=10,
                            checkpoint_dir='checkpoints',
                            experiment_name='srbench_cgp'
                        )
    scaling = True
#cgp_config.init()
else:
    print(f"{args.algo} is not a supported algorithm")
    exit()

for g in group_datasets:
    for d in g:
        print(f"Running dataset: {d}\n")
        X, y = fetch_data(d, return_X_y=True)
        train_X, test_X, train_y, test_y = train_test_split(X, y, train_size=0.75)

        algo = SRBench(args.algo, config, hyperparams, functions=functions, terminals=terminals, scaling_=scaling)
        if args.optimise:
            interface = SMACInterface()

            ## Perform HPO via SMAC
            config.silent_algorithm=True
            config.silent_evolver=True
            trials = 50
            problem  = BlackBox(X, y, algo.loss, 1e-16, True)
            model = TinyTGP(problem, algo.functions, algo.terminals, algo.config, algo.hyperparameters)
            opt_hyperparameters = interface.optimise(model,trials)
            algo = SRBench(args.algo, config, opt_hyperparameters, functions=functions, terminals=terminals, scaling_=scaling)

            
        algo.fit(train_X, train_y) #, checkpoint="examples/checkpoint/srbench_cgp/checkpoint_gen_40.dill")
        print(algo.get_model())
        print(f"{args.algo} train score: {algo.score(train_X, train_y)}")
        print(f"{args.algo} test score: {algo.score(test_X, test_y)}")
        print("="*50)


