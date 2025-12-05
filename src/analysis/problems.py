from src.gp.problem import Problem
from src.gp.tinyverse import GPModel


class Max(Problem):

    def __init__(self, d_):
        self.d = d_
        self.ideal = (4**2)**(self.d - 3)
        self.minimizing = False

    def evaluate(self, genome, model: GPModel) -> int:
        return model.predict(genome = genome, observation = None)[0]