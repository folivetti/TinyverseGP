import math
from abc import ABC

from src.gp.problem import Problem
from src.gp.tinyverse import GPModel


class Max(Problem):

    def evaluate(self, genome, model: GPModel) -> int:
        return model.predict(genome=genome, observation=None)[0]


class MaxPlus(Max):

    def __init__(self, d, t):
        self.ideal = t * 2 ** d
        self.minimizing = False


class MaxPlusMul(Max):

    def __init__(self, d, t):
        self.ideal = math.pow((2 * t), 2 * (d - 1)) if t < 2 else math.pow(t, 2 * (d - 1))
        self.minimizing = False
