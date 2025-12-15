import warnings
from math import sqrt, pi

import numpy
from src.benchmark.policy_search.pl_benchmark import ALEArgs
from src.benchmark.policy_search.plbench.plbench import PLBench
from src.gp.functions import ADD, SUB, MUL, DIV, AND, OR, NAND, NOR, NOTA, IF, LT, GT, BUFA, XOR, XNOR, NOT, EQ, MIN, \
    MAX
from src.gp.problem import PolicySearch
from src.gp.tiny_cgp import CGPHyperparameters, CGPConfig, TinyCGP
from src.gp.tinyverse import Const
import gymnasium as gym

if numpy.version.version[0] == "2":
    warnings.warn("Using NumPy version >=2 can lead to overflow.")

MAXTIME = 3600  # 1 hour
MAXGEN = 1000
LAMBDA = 4
IDEAL = 100000
NUM_EPISODES = 10

functions = [ADD, SUB, MUL, DIV, AND, OR, NAND, NOR, NOT, LT, GT, EQ, MIN, MAX, IF]

config = CGPConfig(
    num_jobs=1,
    max_generations=MAXGEN,
    stopping_criteria=IDEAL,
    minimizing_fitness=False,
    ideal_fitness=IDEAL,
    silent_algorithm=False,
    silent_evolver=False,
    minimalistic_output=True,
    num_functions=len(functions),
    max_arity=2,
    num_inputs=None,
    num_outputs=None,
    report_interval=1,
    max_time=MAXTIME,
    global_seed=42,
    checkpoint_interval=10,
    checkpoint_dir='checkpoint',
    experiment_name='pl_cgp_ale'
)

hyperparameters = CGPHyperparameters(
    mu=1,
    lmbda=LAMBDA,
    population_size=(1+LAMBDA),
    levels_back=100,
    num_function_nodes=100,
    mutation_rate=0.05,
    strict_selection=False,
)

ale_args = ALEArgs(
    noop_max=30,
    frame_skip=1,
    screen_size=32,
    grayscale_obs=True,
    terminal_on_life_loss=True,
    scale_obs=False,
    frame_stack=4,
)

plbench = PLBench(ale_args)
problems = plbench.benchmark["atari_5"]
benchmark = problems["battle_zone"]

env = benchmark.wrapped_env

num_inputs = benchmark.len_observation_space()
num_outputs = benchmark.len_action_space()

config.num_inputs = num_inputs
config.num_outputs = num_outputs

terminals = benchmark.gen_terminals()

problem = PolicySearch(env=env, ideal_=IDEAL, minimizing_=False, num_episodes_=NUM_EPISODES)
cgp = TinyCGP(functions, terminals, config, hyperparameters)
policy = cgp.evolve(problem)
env.close()

env = gym.make("ALE/BattleZone-v5", render_mode="human")
problem = PolicySearch(env=env, ideal_=IDEAL, minimizing_=False)
problem.evaluate(policy.genome, cgp, num_episodes=1, wait_key=True)
env.close()