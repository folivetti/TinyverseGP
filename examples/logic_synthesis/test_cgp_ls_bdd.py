#!/usr/bin/env python3
import os
from src.gp.tiny_cgp import *
from src.gp.problem_bdd import BlackBoxBDD
from src.gp.problem_bdd import  AND, OR, NOT, NAND, NOR, XOR, XNOR, ID

functions = [NOT, ID, AND, OR, XOR, NAND, NOR, XNOR]
#functions = [ID, AND, XOR]
#functions = [NOT, ID, AND, OR, XOR]

# Evolve 3-bit adder
problem = BlackBoxBDD(open(os.path.join(os.path.dirname(__file__),'../../data/logic_synthesis/blif/add3.blif')).read())

# Uncomment to evolve 3-bit adder obtained from the BDD
#import requests
#problem = BlackBoxBDD(requests.get('https://raw.githubusercontent.com/boolean-function-benchmarks/benchmarks/refs/heads/main/benchmarks/blif/add3.blif').text)

config = CGPConfig(
    num_jobs=1,
    max_generations=500_000,
    stopping_criteria=0,
    minimizing_fitness=True,
    ideal_fitness=0,
    silent_algorithm=False,
    silent_evolver=False,
    minimalistic_output=True,
    num_functions=len(functions),
    max_arity=2,
    num_inputs=problem.num_inputs,
    num_outputs=problem.num_outputs,
    num_function_nodes=30,
    report_interval=1000,
    report_every_improvement=True,
    max_time=60000
)

hyperparameters = CGPHyperparameters(
    mu=1,
    lmbda=4,
    population_size=5,
    levels_back=20,
    mutation_rate=0.05,
    #mutation_rate_genes=4,
    strict_selection=False
)

config.init()
random.seed(142)

cgp = TinyCGP(problem, functions, problem.terminals, config, hyperparameters)
best = cgp.evolve()
print('best', best.genome, best.fitness)
print(cgp.evaluate_individual(best.genome))
print('decode', cgp.expression(best.genome))
