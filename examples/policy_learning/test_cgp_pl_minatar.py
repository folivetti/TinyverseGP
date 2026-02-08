from src.benchmark.policy_search.pl_benchmark import PLBenchmark
from src.gp.tiny_cgp import *
from src.gp.problem import PolicySearch
from src.gp.functions import *
from src.gp.tinyverse import Var
import warnings
import numpy

from minatar import gym

gym.register_envs()

if numpy.version.version[0] == "2":
    warnings.warn("Using NumPy version >=2 can lead to overflow.")

NUM_EPISODES = 30
MAX_STEPS = 2e10
GAME = 'breakout'

env = gym.BaseEnv(game=GAME, use_minimal_action_set=True, render_mode="grayscale")
benchmark = PLBenchmark(env, ale_=False, args=None, flatten_obs_=False)
wrapped_env = benchmark.wrapped_env
num_inputs = benchmark.len_observation_space()
num_outputs = benchmark.len_action_space()

functions_ext = [ADD, MUL, DIV, INV, ABS, SIN, COS, TAN, ARCSIN, ARCCOS, ARCTAN, LOG, SQR, SQRT, CEIL, FLOOR,
            AND, OR, NAND, NOR, NOTA, NOTB, BUFA, BUFB, XOR, XNOR, SHFTL, SHFTR,
            LT, GT, EQ, NEQ, MIN, MAX, IF, IFLEZ, IFGTZ]
functions_min = [ADD, MUL, DIV, AND, OR, NAND, NOR, NOT, LT, GT, EQ, MIN, MAX, IF]
terminals = [Var(i) for i in range(num_inputs)]

config = CGPConfig(
    num_jobs=1,
    max_generations=99999999,
    stopping_criteria=100.0,
    minimizing_fitness=False,
    ideal_fitness=100.0,
    silent_algorithm=False,
    silent_evolver=False,
    minimalistic_output=True,
    num_functions=len(functions_ext),
    max_arity=3,
    num_inputs=num_inputs,
    num_outputs=num_outputs,
    report_interval=200,
    max_time=99999999,
    global_seed=42,
    checkpoint_interval=1,
    checkpoint_dir='checkpoint',
    experiment_name='pl_cgp'
)

hyperparameters = CGPHyperparameters(
    mu=1,
    lmbda=1,
    population_size=2,
    num_function_nodes=50,
    levels_back=99999,
    mutation_rate=0.02,
    strict_selection=False,
)

problem = PolicySearch(env=env, ideal_=config.ideal_fitness, minimizing_=False, num_episodes_=NUM_EPISODES,
                       max_steps_=MAX_STEPS)
cgp = TinyCGP(functions_ext, terminals, config, hyperparameters)
policy = cgp.evolve(problem)
env.close()

env = gym.BaseEnv(game=GAME, render_mode="human")
problem = PolicySearch(env=env, ideal_=config.ideal_fitness, minimizing_=False)
problem.evaluate(policy.genome, cgp, num_episodes=1, wait_key=True)
env.close()
