import sys

from src.analysis.benchmarks.boolean import Conjunction, NegVar
from src.analysis.models.simple_cgp import SimpleCGP, SimpleCGPConfig, MutationType
from src.gp.tiny_cgp import *
from src.gp.functions import AND

MAX_GENERATIONS = 1000000
MAX_TIME = 9999999
N = int(sys.argv[1])
MAX_ARITY = 2
NUM_GENES = (MAX_ARITY + 1) * N + 1
MUTATION_RATE = 1 / NUM_GENES
USE_NEGATED_VARIABLES = False
USE_COMPLETE_TRAINING_SET = True

functions = [AND]
terminals = [Var(i) for i in range(N)]

if USE_NEGATED_VARIABLES:
    terminals += [NegVar(i) for i in range(N)]

config = SimpleCGPConfig(
    num_jobs=1,
    max_generations=MAX_GENERATIONS,
    stopping_criteria=0,
    minimizing_fitness=True,
    ideal_fitness=0,
    silent_algorithm=True,
    silent_evolver=True,
    minimalistic_output=True,
    num_functions=len(functions),
    max_arity=2,
    num_inputs=len(terminals),
    num_outputs=1,
    report_interval=1,
    max_time=MAX_TIME,
    mutation_type=MutationType.SAM,
    global_seed=None,
    checkpoint_interval=9999999,
    checkpoint_dir='../checkpoint',
    experiment_name='and_cgp'
)

hyperparameters = CGPHyperparameters(
    mu=1,
    lmbda=1,
    population_size=2,
    num_function_nodes=N + 1,
    levels_back=N + 1,
    mutation_rate=MUTATION_RATE,
    strict_selection=False
)

problem = Conjunction(n=N, use_complete_training_set=USE_COMPLETE_TRAINING_SET)
config.ideal_fitness = problem.ideal
config.global_seed = int(time.time_ns())
cgp = SimpleCGP(functions, terminals, config, hyperparameters)
program = cgp.evolve(problem)

print(f"{N},simple_cgp,{cgp.generation_number},{problem.calc_generalization_error(program.genome, cgp)}")
