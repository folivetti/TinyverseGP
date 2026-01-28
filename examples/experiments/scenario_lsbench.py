from src.benchmark.logic_synthesis.lsbench.lsbench import LSBench, LSRegressor
from src.gp.tiny_cgp import CGPConfig, CGPHyperparameters
from src.gp.tiny_ge import GEHyperparameters
from src.gp.tiny_lgp import LGPConfig, LGPHyperparameters
from src.gp.tiny_tgp import TGPConfig, TGPHyperparameters
from src.gp.tinyverse import Var, GPConfig
import argparse
import csv
import pathlib


parser = argparse.ArgumentParser(
                                prog='LSBench Test Scenario',
                                description='Run algorithm on selected LSBench instances',
                                epilog='')

parser.add_argument('--maxtime', dest='maxtime', type=int, default=3600)
parser.add_argument('--maxgen', dest='maxgen', type=int, default=100)
parser.add_argument('--popsize', dest='popsize', type=int, default=100)
parser.add_argument('--algo', dest='algo', type=str, default="TGP")
parser.add_argument('-d', '--dataset', dest='dataset', type=str, default="")
parser.add_argument('-s', '--seed', dest='seed', type=int, default=42)
parser.add_argument('-o', '--optimise', action='store_true')
parser.add_argument('-v', '--verbose', action='store_true')  # on/off flag

lsbench = LSBench(data_dir_=pathlib.Path(__file__).parent.parent.parent.resolve() / 'data' / 'logic_synthesis')
if args.dataset != "":
    benchmarks = [getattr(lsbench, args.dataset)()]
else:
    benchmarks = [lsbench.add4(),
                    lsbench.mul3(),
                    lsbench.alu4(),
                    lsbench.count4(),
                    lsbench.dec4(),
                    lsbench.enc8(),
                    lsbench.epar8(),
                    lsbench.mcomp4(),
                    lsbench.icomp5()
                    ]



functions_reduced = ["AND", "OR", "BUFA", "NOT"]
functions_extended = ["AND", "OR", "BUFA", "NOT", "XOR", "NAND", "NOR", "XNOR"]
terminals = None

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
        population_size=args.popsize,
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
    hyperparams = LGPHyperparameters(
                        mu=args.popsize,
                        macro_variation_rate=0.75,
                        micro_variation_rate=0.25,
                        insertion_rate=0.5,
                        max_segment=25,
                        reproduction_rate=0.5,
                        branch_probability=0.1,
                        p_register = 0.5,
                        max_len = 500,
                        initial_max_len = 35,
                        erc = False,
                        default_value = 0.0,
                        protection = 1e10,
                        penalization_validity_factor=0.0
                    )
    config = LGPConfig(
        num_jobs=1,
        max_generations=args.maxgen,
        stopping_criteria=1e-6,
        minimizing_fitness=True,
        ideal_fitness=1e-6,
        silent_algorithm=True,
        silent_evolver=True,
        minimalistic_output=True,
        report_interval=100000000000,
        max_time=args.maxtime,
        num_registers=8,
        global_seed=args.seed,
        checkpoint_interval=10,
        checkpoint_dir="checkpoint",
        experiment_name="srbench_lgp",
    )
    scaling = True
    
elif args.algo == "GE":
    hyperparams = GEHyperparameters(
        pop_size=args.popsize,
        genome_length=40,
        codon_size=1000,
        cx_rate=0.9,
        mutation_rate=0.1,
        tournament_size=2,
        penalty_value=99999,
    )
    
    config = GPConfig(
        num_jobs=1,
        max_generations=args.maxgen,
        stopping_criteria=1e-6,
        minimizing_fitness=True,  # this should be used from the problem instance
        ideal_fitness=1e-6,  # this should be used from the problem instance
        silent_algorithm=False,
        silent_evolver=False,
        minimalistic_output=True,
        num_outputs=1,
        report_interval=1,
        max_time=args.maxtime,
        global_seed=args.seed,
        checkpoint_interval=10,
        checkpoint_dir="checkpoint",
        experiment_name="srbench_ge",
    )
    scaling = True
#cgp_config.init()
else:
    print(f"{args.algo} is not a supported algorithm")
    exit()

# print("LSBench has been created!")

for bm in benchmarks:
    csv_filename = f"experiments_scripts/output_data/{d}_{args.algo}_{args.seed}.csv"
    with open(csv_filename, mode="w", newline="") as csv_file:
        fieldnames = ["dataset_name", "algo_name", "nb_trials", "def_parameters", "opt_parameters", "default", "optimised", "seed"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        print(f"Running benchmark: {bm.name}")

        num_inputs = bm.benchmark.num_inputs
        num_outputs = bm.benchmark.num_outputs

        if algo in ['TGP', 'CGP', 'LGP', 'GE']:
            config.num_outputs = num_outputs
            
            if args.algo in ['TGP', 'CGP', 'LGP']:
                config.num_inputs = num_inputs
                if args.algo == 'LGP':
                    config.num_registers = num_outputs

        tt = bm.get_truth_table()

        functions = lsbench.get_fs(bm.name)
        terminals = [Var(index=i, name_="x" + str(+ i)) for i in range(num_inputs)]

        algo = LSRegressor(
            representation_=args.algo,
            config_=config,
            hyperparameters_=hyperparams,
            functions_=functions,
            terminals_=terminals
        )

        algo.fit(X=tt.inputs, y=tt.outputs)
        print(f"{args.algo} score: {algo.score(tt.inputs, tt.outputs)}")
        break
