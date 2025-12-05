import math

from src.gp.problem import Problem
from src.gp.tinyverse import GPModel


class Max(Problem):

    def __init__(self, d, t):
        self.ideal = math.pow(4,2*(d-3))
        self.minimizing = False

    def evaluate(self, genome, model: GPModel) -> int:
        return model.predict(genome = genome, observation = None)[0]