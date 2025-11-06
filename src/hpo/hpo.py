from abc import ABC, abstractmethod
from src.gp.problem import Problem
from src.gp.tinyverse import GPModel, GPHyperparameters
from ConfigSpace import Configuration, ConfigurationSpace
from smac import HyperparameterOptimizationFacade, Scenario
import copy

class HPOInterface(ABC):
    """
    Abstract class that is used to implement methods for running HPO with GP models
    that are provided within TinverseGP.
    """

    @abstractmethod
    def optimise(self, gpmodel_: GPModel, *params):
        pass


class SMACInterface(HPOInterface):


    def optimise(self, gpmodel_: GPModel, problem:Problem, n_trials_:int=10, seed_:int=42) -> GPHyperparameters:
        """Optimises the hyperparameters of a GP model using SMAC. (https://github.com/automl/SMAC3)

        Args:
            gpmodel_ (GPModel): the GP model to be optimised
            problem (Problem): _description_
            n_trials_ (int, optional): Number of trials in the SMAC optimisation environment. Defaults to 10.
            seed_ (int, optional): used by SMAC. Defaults to 42.

        Returns:
            GPHyperparameters: the hyperparameter configuration to be used
        """

        def train(config: Configuration, seed: int = 0) -> float:
            gpmodel = copy.deepcopy(gpmodel_)
            # Apply hyperparameters
            for c in config.keys():
                setattr(gpmodel.hyperparameters, c, config[c])
            # Randomize seed for stochasticity
            # gpmodel.hyperparameters.global_seed = randrange(0, 100000)
            # gpmodel.hyperparameters.global_seed = seed
            print("Config applied:", {key: getattr(gpmodel.hyperparameters, key) for key in config.keys()})
            gpmodel.evolve(problem)
            fitness = gpmodel.best_individual.fitness
            print("Fitness obtained:", fitness)
            return fitness

        # Configuration space
        configspace = ConfigurationSpace(gpmodel_.hyperparameters.space)

        # SMAC scenario (stochastic)
        scenario = Scenario(configspace, deterministic=True, n_trials=n_trials_, seed=seed_)

        # Run SMAC
        smac = HyperparameterOptimizationFacade(scenario, train, overwrite=True)
        incumbent = smac.optimize()

        # Map incumbent to hyperparameters object
        inc_hp = copy.deepcopy(gpmodel_.hyperparameters)
        for c in incumbent.keys():
            setattr(inc_hp, c, incumbent[c])

        return inc_hp