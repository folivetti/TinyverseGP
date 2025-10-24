"""
Example module to test 3GE with program synthesis problems.

Attempts to evolve a solution for the numbers of two problem that
is provided on Leetcode.com:

https://leetcode.com/problems/power-of-two/description/

"""

import warnings

warnings.filterwarnings("ignore")

from src.gp.problem import ProgramSynthesis
from src.benchmark.program_synthesis.ps_benchmark import PSBenchmark
from src.benchmark.program_synthesis.leetcode.power_of_two import *
from src.gp.functions import *
from src.gp.tiny_3GE import *


config = GPConfig(
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

hyperparameters = TreeGEHyperparameters(
    pop_size=16,
    min_depth=4,
    max_depth=6,
    codon_size=256,
    cx_rate=0.9,
    mutation_rate=0.1,
    tournament_size=2,
    penalty_value=99999,
)

generator = gen_power_of_two
n = 10
m = 100

benchmark = PSBenchmark(generator, [n, m])
problem = ProgramSynthesis(benchmark.dataset)

functions = [ADD, SUB, MUL, DIV, AND, OR, NAND, NOR, NOT, IF, LT, GT]
arguments = ["x"]
grammar = {
    "<expr>": [
        "ADD(<expr>, <expr>)",
        "SUB(<expr>, <expr>)",
        "MUL(<expr>, <expr>)",
        "DIV(<expr>, <expr>)",
        "AND(<expr>, <expr>)",
        "OR(<expr>, <expr>)",
        "NAND(<expr>, <expr>)",
        "NOR(<expr>, <expr>)",
        "NOT(<expr>)",
        "IF(<expr>, <expr>, <expr>)",
        "LT(<expr>, <expr>)",
        "GT(<expr>, <expr>)",
        "<d>",
        "<d>.<d><d>",
        "x",
    ],
    "<d>": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"],
}

tree_ge = Tiny3GE(functions, grammar, arguments, config, hyperparameters)
tree_ge.print_population(tree_ge.population)

tree_ge.evolve(problem)

tree_ge.print_individual_tree(tree_ge.best_individual.deriv_tree)
tree_ge.print_individual(tree_ge.best_individual)