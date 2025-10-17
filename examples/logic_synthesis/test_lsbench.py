from src.benchmark.logic_synthesis.lsbench.lsbench import LSBench
from src.gp.tiny_cgp import CGPConfig, CGPHyperparameters
from src.gp.tiny_tgp import TGPConfig, TGPHyperparameters
from src.gp.tinyverse import Var

MAXTIME = 3600  # 1 hour
MAXGEN = 30
POPSIZE = 20

functions=["AND", "OR", "BUFA", "NOT"]
terminals = None

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
    strict_selection=True,
    mutation_rate=0.3,
    population_size=POPSIZE,
    levels_back=10,
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
    report_interval=10,
    max_time=MAXTIME,
    global_seed=42,
    checkpoint_interval=10,
    checkpoint_dir="examples/checkpoint",
    experiment_name="srbench_cgp",
)

lsbench = LSBench(data_dir_='../../data/logic_synthesis')

tgp = LSBench.LSRegressor(
            representation_="TGP",
            config_=tgp_config,
            hyperparameters_=tgp_hyperparams,
            functions_=functions,
            terminals_=terminals
        )

cgp = LSBench.LSRegressor(
             "CGP",
             cgp_config,
             cgp_hyperparams,
             functions_=functions,
             terminals_=terminals,
        )

for k in lsbench.benchmarks:
    bm = lsbench.benchmarks[k]
    num_inputs = bm.benchmark.num_inputs
    tgp_config.num_inputs = num_inputs
    tgp_config.num_outputs = num_outputs
    tt = bm.get_truth_table()
    terminals = [Var(i) for i in range(num_inputs)]

    tgp.terminals = terminals
    tgp.fit(X=tt.inputs, y=tt.outputs)

for k in lsbench.benchmarks:
    print(k)
    bm = lsbench.benchmarks[k]
    num_inputs = bm.benchmark.num_inputs
    num_outputs = bm.benchmark.num_outputs
    cgp_config.num_inputs = num_inputs
    cgp_config.num_outputs = num_outputs
    tt = bm.get_truth_table()
    terminals = [Var(i) for i in range(num_inputs)]

    cgp.terminals = terminals
    cgp.fit(X=tt.inputs, y=tt.outputs)
