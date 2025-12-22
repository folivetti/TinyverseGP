from math import log2, pow, exp2
from src.analysis.log_transform import transform_log
from src.gp.problem import Problem
from src.gp.tinyverse import GPModel


class Max(Problem):
    log_transform: bool

    def __init__(self, t, log_scaling_ = False):
        assert(t > 0)
        self.log_transform = log_scaling_


    def evaluate(self, genome, model: GPModel, scale_log_ = False) -> int:
        return model.predict(genome=genome, observation=None)[0]


class MaxPlus(Max):

    def __init__(self, d, t):
        super().__init__(t=t, log_scaling_=False)
        self.ideal = t * pow(2, d)
        self.minimizing = False


class MaxPlusMul(Max):

    def __init__(self, d, t, log_scaling = False):
        super().__init__(t = t, log_scaling_=log_scaling)
        if log_scaling:
            self.ideal = exp2(d-1) * log2(max(2*t, t*t))
        else:
            self.ideal = pow(max(2 * t, t * t), pow(2, d - 1))
        self.minimizing = False

