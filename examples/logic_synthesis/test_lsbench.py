from src.benchmark.logic_synthesis.lsbench.lsbench import LSBench, LSRegressor
from src.gp.tiny_cgp import CGPConfig, CGPHyperparameters
from src.gp.tiny_tgp import TGPConfig, TGPHyperparameters
from src.gp.tinyverse import Var

MAXTIME = 3600  # 1 hour
MAXGEN = 50
POPSIZE = 50

lsbench = LSBench(data_dir_='../../data/logic_synthesis')
benchmarks = [lsbench.add3(),
              lsbench.mul3(),
              lsbench.alu3(),
              lsbench.count4(),
              lsbench.dec4(),
              lsbench.enc8(),
              lsbench.epar8(),
              lsbench.mcomp3(),
              lsbench.icomp5()]

functions=["AND", "OR", "BUFA", "NOT"]
terminals = None

tgp_hyperparams = TGPHyperparameters(
    max_depth=8,
    max_size=100,
    pop_size=POPSIZE,
    tournament_size=3,
    mutation_rate=0.2,
    cx_rate=0.9,
    erc = None
)

cgp_hyperparams = CGPHyperparameters(
    mu=1,
    lmbda=4,
    num_function_nodes=10,
    strict_selection=True,
    mutation_rate=0.2,
    population_size=POPSIZE,
    levels_back=10,
)

#   Set up configurations for TGP and CGP
tgp_config = TGPConfig(
    num_jobs=1,
    max_generations=MAXGEN,
    stopping_criteria=0,
    minimizing_fitness=True,
    ideal_fitness=0,
    silent_algorithm=True,
    silent_evolver=True,
    minimalistic_output=True,
    num_outputs=1,
    report_interval=1000000,
    max_time=MAXTIME,
    global_seed=42,
    checkpoint_interval=10,
    checkpoint_dir="examples/checkpoint",
    experiment_name="lsbench_tgp",
)

cgp_config = CGPConfig(
    num_jobs=1,
    max_generations=MAXGEN,
    stopping_criteria=0,
    minimizing_fitness=True,
    ideal_fitness=0,
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
    experiment_name="lsbench_cgp",
)

#print("LSBench has been created!")

for bm in benchmarks:

    print(f"Running benchmark: {bm.name}\n")

    num_inputs = bm.benchmark.num_inputs
    num_outputs = bm.benchmark.num_outputs

    tgp_config.num_inputs = num_inputs
    tgp_config.num_outputs = num_outputs

    cgp_config.num_inputs = num_inputs
    cgp_config.num_outputs = num_outputs

    tgp = LSRegressor(
        representation_="TGP",
        config_=tgp_config,
        hyperparameters_=tgp_hyperparams,
        functions_=functions,
        terminals_=terminals
    )

    cgp = LSRegressor(
        "CGP",
        cgp_config,
        cgp_hyperparams,
        functions_=functions,
        terminals_=terminals,
    )

    tt = bm.get_truth_table()

    terminals = [Var(i) for i in range(num_inputs)]

    tgp.terminals = terminals
    tgp.fit(X=tt.inputs, y=tt.outputs)

    print(f"tgp score: {tgp.score(tt.inputs,tt.outputs)}")

    cgp.terminals = terminals
    cgp.fit(X=tt.inputs, y=tt.outputs)

    print(f"cgp score: {cgp.score(tt.inputs, tt.outputs)}")
