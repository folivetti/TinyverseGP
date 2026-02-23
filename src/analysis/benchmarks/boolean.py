from src.gp.problem import Problem
import numpy as np
from functools import reduce

class BooleanFunction(Problem):
    n_in: int
    n_out: int
    training_set: np.array
    operator: callable

    def __init__(self, n_in, n_out, operator: callable):
        self.n_in = n_in
        self.n_out = n_out
        self.operator = operator
        self.init_training_set()

    def init_variables(self) -> np.array:
        rows = int(pow(2, self.n_in))
        cols = self.n_in + 1
        vars = cols - self.n_out
        training_set = np.zeros(shape=(rows, cols), dtype=np.uint8)

        for c in range(vars):
            d = pow(2, vars - c)
            for r in range(rows):
                if r % d >= d / 2:
                    training_set[r][c] = 1
        return rows, cols, training_set

    def random_training_subset(self, n) -> np.array:
        rand_indices = np.random.choice(self.training_set.shape[0],
                         size=n,
                         replace=False)
        return self.training_set[rand_indices, :]

    def init_training_set(self):
        rows, cols, training_set = self.init_variables()

        for row in training_set:
            args = row[0:cols - self.n_out]
            res = reduce(self.operator, args)
            row[cols - 1] = res
        self.training_set = training_set


class Conjunction(BooleanFunction):
    def __init__(self, n):
        super().__init__(n_in = n, n_out=1, operator=lambda x, y: x & y)


class ExclusiveDisjunction(BooleanFunction):
    def __init__(self, n):
        super().__init__(n_in=n, n_out=1, operator=lambda x, y: x ^ y)

and_ = Conjunction(n=4)
print(and_.random_training_subset(6))