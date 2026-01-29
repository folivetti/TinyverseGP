from abc import ABC, abstractmethod
from src.gp.problem import Problem
from src.gp.tinyverse import GPModel, GPHyperparameters
from src.gp.problem import Problem
from ConfigSpace import Configuration, ConfigurationSpace
from smac import HyperparameterOptimizationFacade, Scenario
import copy
from sklearn.base import RegressorMixin
import numpy as np

class HPOInterface(ABC):
    """
    Abstract class that is used to implement methods for running HPO with GP models
    that are provided within TinverseGP.
    """

    @abstractmethod
    def optimise(self, gpmodel_: GPModel, *params):
        pass


class SMACInterface(HPOInterface):
    def optimise(self, gpmodel_: GPModel, problem : Problem, n_trials_=10) -> GPHyperparameters:
        """
        Runs HPO with SMAC (https://github.com/automl/SMAC3)
        Args:
            gpmodel_ (GPModel): the GP model to be optimised
            n_trials_: Number of trials in the SMAC optimisation environment
            train_X: Training features
            train_y: Training labels
        Returns:
            GPHyperparameters: the hyperparameter configuration to be used
        """
        def train(config: Configuration, seed: int = 0) -> float:
            gpmodel = copy.deepcopy(gpmodel_)
            # Apply hyperparameters
            for c in config.keys():
                setattr(gpmodel.hyperparameters, c, config[c])
            # Use train_X and train_y in the model's training process
            gpmodel.evolve(problem)
            return gpmodel.best_individual.fitness

        # Obtain the hyperparameter (HP) space from the GP model
        paramspace = gpmodel_.hyperparameters.space
        # Use the HP space to init the configuration space (CS)
        configspace = ConfigurationSpace(paramspace)
        # Scenario object specifying the optimization environment
        scenario = Scenario(configspace, deterministic=True, n_trials=n_trials_)
        # Use SMAC to find the best configuration/hyperparameters
        smac = HyperparameterOptimizationFacade(scenario, train)
        incumbent = smac.optimize()

        # Map incumbent to hyperparameters object
        inc_hp = copy.deepcopy(gpmodel_.hyperparameters)
        for c in incumbent.keys():
            setattr(inc_hp, c, incumbent[c])

class SMAC4BenchInterface:
    def __init__(self, bench: RegressorMixin):
        """
        Initialize the SMAC4SRBenchInterface with an SRBench instance.
        Args:
            bench: An instance of RegressorMixin to be optimized.
        """
        self.bench = bench
        self.model_type = bench.representation  # 'TGP' or 'CGP'

    def optimise(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_trials: int = 10,
        seed: int = 42,
        sc_name: str = "default",
        fn_eval_limit: int = -1,
        fn_eval_per_gen: str = "pop_size"
    ) -> GPHyperparameters:
        """
        Optimize the hyperparameters of the SRBench model using SMAC.
        Args:
            X: Training features.
            y: Training labels.
            n_trials: Number of SMAC trials.
            seed: Random seed for reproducibility.
            sc_name: Scenario name for SMAC output directory.
            fn_eval_limit: Target function evaluation limit (override generations).
            fn_eval_per_gen: Target function evaluation per generation.
        Returns:
            Dictionary of optimized hyperparameters.
        """

        def train(config: Configuration, seed: int = 0) -> float:
            # Create a copy of the SRBench instance
            bench = copy.deepcopy(self.bench)
            # Apply the SMAC configuration to the hyperparameters
            for param, value in config.items():
                setattr(bench.hyperparameters, param, value)
            # adapt function evaluation limit if applicable
            if fn_eval_limit > 0:
                if fn_eval_per_gen == 'pop_size':
                    bench.hyperparameters.max_gen = fn_eval_limit // bench.config.pop_size
                elif fn_eval_per_gen == 'lambda':
                    bench.hyperparameters.max_gen = fn_eval_limit // bench.hyperparameters.lmbda
                elif fn_eval_per_gen.isdigit():
                    bench.hyperparameters.max_gen = fn_eval_limit // int(fn_eval_per_gen)
                else:
                    raise ValueError(f"Unknown fn_eval_per_gen: {fn_eval_per_gen}")
            # Fit the model
            bench.fit(X, y)
            # Return the fitness (e.g., negative MSE for minimization)
            return -bench.score(X, y)  # SMAC minimizes, so return negative score

        # Get the hyperparameter space from the SRBench's hyperparameters
        paramspace = self.bench.hyperparameters.space
        # take out max_gen if function eval limit is set
        if fn_eval_limit > 0:
            if 'max_gen' in paramspace:
                del paramspace['max_gen']
        # Initialize the configuration space
        configspace = ConfigurationSpace(paramspace)
        # Define the SMAC scenario
        output_dir = f"experiments_scripts/smac3-output_{sc_name}_{seed}"
        scenario = Scenario(
            configspace,
            deterministic=True,
            n_trials=n_trials,
            seed=seed,
            output_directory=output_dir
        )
        # Run SMAC optimization
        smac = HyperparameterOptimizationFacade(scenario, train)
        incumbent = smac.optimize()
        inc_hp = copy.deepcopy(self.bench.hyperparameters)
        for c in incumbent.keys():
            setattr(inc_hp, c, incumbent[c])
        return inc_hp
    
class Hpo:
    """
    Class that provides methods to run HPO for GP models
    that are provided within TinverseGP.
    """

    def optimise_smac(gpmodel_: GPModel, n_trials_=10) -> GPHyperparameters:
        """
        Runs HPO with SMAC.

        Args:
            gpmodel_ (GPModel): the model to be optimised

        Returns:
            GPHyperparameters: the hyperparameter configuration to be used
        """

        def train(config: Configuration, seed: int = 0) -> float:
            gpmodel = copy.deepcopy(gpmodel_)
            for c in config.keys():
                setattr(gpmodel.hyperparameters, c, config[c])
            gpmodel.evolve()
            return gpmodel.best_individual.fitness

        # Obtain the hyperparameter (HP) space from the GP model
        space = gpmodel_.hyperparameters.space

        # Use the HP space to init the configuration space (CS)
        cs = ConfigurationSpace(space)

        # Scenario object specifying the optimization environment
        scenario = Scenario(cs, deterministic=True, n_trials=n_trials_)

        # Use SMAC to find the best configuration/hyperparameters
        smac = HyperparameterOptimizationFacade(scenario, train)
        incumbent = smac.optimize()
        incHP = copy.deepcopy(gpmodel_.hyperparameters)
        for c in incumbent.keys():
            setattr(incHP, c, incumbent[c])
        return incHP
