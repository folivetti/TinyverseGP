from abc import ABC, abstractmethod
from src.gp.tinyverse import GPModel, GPHyperparameters
from ConfigSpace import Configuration, ConfigurationSpace
from smac import HyperparameterOptimizationFacade, Scenario
import copy
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from src.gp.problem import BlackBox
from src.gp.loss import *
from src.gp.tiny_cgp import *

class HPOInterface(ABC):
    """
    Abstract class that is used to implement methods for running HPO with GP models
    that are provided within TinverseGP.
    """

    @abstractmethod
    def optimise(self, gpmodel_: GPModel, *params):
        pass


class SMACInterface(HPOInterface):

    def optimise(self, gpmodel_: GPModel, dataset_: str, train_X_, train_y_, n_trials_=10, seed_=42, n_splits_ = 5) -> GPHyperparameters:
        """
        Runs HPO with SMAC (https://github.com/automl/SMAC3)

        Args:
            gpmodel_ (GPModel): the GP model to be optimised
            n_trials_: Number of trials in the SMAC optimisation environment

        Returns:
            GPHyperparameters: the hyperparameter configuration to be used
        """

        def train(config: Configuration, seed: int = 0) -> float:
            gpmodel = copy.deepcopy(gpmodel_)
            for c in config.keys():
                setattr(gpmodel.hyperparameters, c, config[c])
            print("Config applied:", {key: getattr(gpmodel.hyperparameters, key) for key in config.keys()})

            # Calculate mean train score using k-fold CV
            fitness = []
            kf = KFold(n_splits=n_splits_, shuffle=True, random_state=12345)

            for fold, (train_index, test_index) in enumerate(kf.split(X=train_X_, y=train_y_)):
                # Train on train set (from original train set that has been further split into train and test sets)
                gpmodel = copy.deepcopy(gpmodel_)
                for c in config.keys():
                    setattr(gpmodel.hyperparameters, c, config[c])
                train_X = train_X_[train_index]
                train_y = train_y_[train_index]
                gpmodel.problem  = BlackBox(train_X, train_y, mean_squared_error, 1e-16, True)
                gpmodel.evolve()
                
                # Evaluate on test set
                test_X = train_X_[test_index]
                test_y = train_y_[test_index]
                best_individual_genome = gpmodel.best_individual.genome
                problem  = BlackBox(test_X, test_y, mean_squared_error, 1e-16, True)
                test_fitness = problem.evaluate(best_individual_genome, gpmodel)

                fitness.append(test_fitness)

            mean_fitness = np.mean(fitness)
            print(f"Mean fitness: {mean_fitness}")

            return mean_fitness

        # Obtain the hyperparameter (HP) space from the GP model
        paramspace = gpmodel_.hyperparameters.space

        # Use the HP space to init the configuration space (CS)
        configspace = ConfigurationSpace(paramspace)

        # Scenario object specifying the optimization environment
        scenario = Scenario(configspace, name=f"{gpmodel_.hyperparameters.operator}_{n_trials_}_{dataset_}", deterministic=True, n_trials=n_trials_, seed=seed_)

        # Use SMAC to find the best configuration/hyperparameters
        smac = HyperparameterOptimizationFacade(scenario, train, overwrite=False)
        incumbent = smac.optimize()

        inc_hp = copy.deepcopy(gpmodel_.hyperparameters)
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
