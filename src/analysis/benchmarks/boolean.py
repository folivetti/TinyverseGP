from src.gp.loss import hamming_distance_bitwise
from src.gp.problem import BlackBox
import numpy as np
from functools import reduce


class BooleanFunction(BlackBox):
    n_in: int
    n_out: int
    training_set: np.array
    operator: callable

    def __init__(self, n_in, n_out, operator: callable):

        rows, cols, training_set = BooleanFunction.init_training_set(n_in, n_out, operator)
        observations = training_set[:, : cols - 1]
        actual = training_set[:, cols - 1]
        loss = hamming_distance_bitwise

        super().__init__(actual_=actual,
                         observations_=observations,
                         loss_=loss,
                         ideal_=0,
                         minimizing_=True)
        self.n_in = n_in
        self.n_out = n_out
        self.operator = operator
        self.training_set = training_set

    @staticmethod
    def init_variables(n_in, n_out) -> np.array:
        rows = int(pow(2, n_in))
        cols = n_in + 1
        vars = cols - n_out
        training_set = np.zeros(shape=(rows, cols), dtype=np.uint8)

        for c in range(vars):
            d = pow(2, vars - c)
            for r in range(rows):
                if r % d >= d / 2:
                    training_set[r][c] = 1
        return rows, cols, training_set

    @staticmethod
    def init_training_set(n_in, n_out, operator) -> np.array:
        rows, cols, training_set = BooleanFunction.init_variables(n_in, n_out)

        for row in training_set:
            args = row[0:cols - n_out]
            res = reduce(operator, args)
            row[cols - 1] = res
        return rows, cols, training_set

    def random_training_subset(self, n) -> np.array:
        rand_indices = np.random.choice(self.training_set.shape[0],
                                        size=n,
                                        replace=False)
        return self.training_set[rand_indices, :]


class Conjunction(BooleanFunction):
    def __init__(self, n):
        super().__init__(n_in=n, n_out=1, operator=lambda x, y: x & y)


class ExclusiveDisjunction(BooleanFunction):
    def __init__(self, n):
        super().__init__(n_in=n, n_out=1, operator=lambda x, y: x ^ y)
