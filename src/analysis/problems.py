import math

from src.gp.problem import Problem
from src.gp.tinyverse import GPModel


class Max(Problem):

    def evaluate(self, genome, model: GPModel) -> int:
        return model.predict(genome=genome, observation=None)[0]


class MaxPlus(Max):

    def __init__(self, d, t):
        self.ideal = t * math.pow(2,d)
        self.minimizing = False


class MaxPlusMul(Max):

    def __init__(self, d, t):
        self.ideal = max(math.pow(2 * t,t*t), math.pow(2, (d - 1)))
        self.minimizing = False
