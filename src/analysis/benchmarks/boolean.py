from math import pow, ceil
from dataclasses import dataclass
from typing_extensions import override
from src.gp.loss import hamming_distance_bitwise
from src.gp.problem import BlackBox
from functools import reduce
from src.gp.tinyverse import Var
import numpy as np


class NegVar(Var):

    def __init__(self, index: int = None, name_: str = None):
        super().__init__(index, name_="NegVar")

    @override
    def __call__(self, *args, **kwargs):
        return ~(super().__call__())


@dataclass
class TrainingSet:
    data: np.array
    rows: int
    cols: int

    def __len__(self):
        return self.rows

    def get_observations(self):
        return self.data[:, : self.cols - 1]

    def get_actual(self):
        return self.data[:, self.cols - 1]


class BooleanFunction(BlackBox):
    n_in: int
    n_out: int
    training_set: TrainingSet
    operator: callable
    use_complete_training_set: bool
    training_set_size: int
    k: float

    def __init__(self, n_in: int, n_out: int, operator: callable, use_complete_training_set: bool = True,
                 negated_vars = False, k: float = 1.5):

        training_set = BooleanFunction.init_training_set(n_in, n_out, operator, negated_vars)
        super().__init__(actual_=training_set.get_actual(),
                         observations_=training_set.get_observations(),
                         loss_=hamming_distance_bitwise,
                         ideal_=0,
                         minimizing_=True)
        self.n_in = n_in
        self.n_out = n_out
        self.operator = operator
        self.training_set = training_set
        self.use_complete_training_set = use_complete_training_set
        self.negated_vars = negated_vars
        self.k = k

        if use_complete_training_set:
            self.training_set_size = len(self.training_set)
        else:
            self.training_set_size = ceil(pow(self.n_in, self.k))

    @staticmethod
    def init_variables(n_in, n_out, negated_vars = False) -> np.array:
        rows = int(pow(2, n_in))
        if not negated_vars:
            cols = n_in + 1
        else:
            cols = (2 * n_in) + 1
        n_vars = n_in
        training_set = np.zeros(shape=(rows, cols), dtype=np.int8)

        for c in range(n_vars):
            d = pow(2, n_vars - c)
            for r in range(rows):
                if r % d >= d / 2:
                    training_set[r][c] = 1

        if negated_vars:
            for r in range(rows):
                for c in range(n_vars):
                    training_set[r][n_vars + c] = 1 if training_set[r][c] == 0 else 0

        return rows, cols, training_set

    @staticmethod
    def init_training_set(n_in, n_out, operator, negated_vars = False) -> TrainingSet:
        rows, cols, training_set = BooleanFunction.init_variables(n_in, n_out, negated_vars)

        for row in training_set:
            args = row[0:n_in]
            res = reduce(operator, args)
            row[cols - 1] = res

        return TrainingSet(data=training_set, cols=cols, rows=rows)

    def sample_training_subset(self, s) -> np.array:
        rand_indices = np.random.choice(self.training_set.rows,
                                        size=s,
                                        replace=False)
        return self.training_set.data[rand_indices, :]

    def calc_generalization_error(self, program, model) -> float:
        self.observations, self.actual = self.get_training_set(complete=True)
        return self.evaluate(program, model)

    def get_training_set(self, complete = None) -> tuple:
        if complete or self.use_complete_training_set:
            return self.training_set.get_observations(), self.training_set.get_actual()
        else:
            training_subset = self.sample_training_subset(self.training_set_size)
            return training_subset[:, : self.training_set.cols - 1], training_subset[:, self.training_set.cols - 1]

    @override
    def cost(self, predictions: list) -> float:
        self.observations, self.actual = self.get_training_set()
        return super().cost(predictions)


class Conjunction(BooleanFunction):
    def __init__(self, n, use_complete_training_set=True, negated_vars = False):
        super().__init__(n_in=n, n_out=1, operator=lambda x, y: x & y,
                         use_complete_training_set=use_complete_training_set,
                         negated_vars=negated_vars)


class ExclusiveDisjunction(BooleanFunction):
    def __init__(self, n, use_complete_training_set=True, negated_vars = False):
        super().__init__(n_in=n, n_out=1, operator=lambda x, y: x ^ y,
                         use_complete_training_set=use_complete_training_set,
                         negated_vars=negated_vars)
