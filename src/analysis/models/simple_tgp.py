import copy
import random
from dataclasses import dataclass
from typing import override
from src.gp.tiny_tgp import TGPIndividual, Node, TinyTGP, TGPConfig
from src.gp.tinyverse import Var, Const, Hyperparameters


@dataclass(kw_only=True)
class SGPHyperparameters(Hyperparameters):
    lmbda: int = 1
    k: int = 1
    strict_selection: bool = False


class HVLPrime:

    def __init__(self, functions_: list, terminals_: list):
        self.functions = functions_
        self.terminals = terminals_

    def is_leaf(self, n: Node):
        return isinstance(n.function, Var) or isinstance(n.function, Const)

    def rnd_leaf(self, n: Node, p=None) -> tuple[Node, Node]:
        if self.is_leaf(n):
            return n, p
        return self.rnd_leaf(random.choice(n.children), n)

    def rnd_inner(self, n: Node, p: float) -> Node:
        if random.random() <= p:
            return n

        c = [child for child in n.children if self.is_leaf(child) == False]

        if len(c) > 0:
            n = self.rnd_inner(random.choice(c), p)
        return n

    def count_inner_nodes(self, node: Node) -> int:
        """
        Return the number of inner nodes in a tree.
        """
        if len(node.children) == 0:
            return 0
        s = 0
        for child in node.children:
            if child.function is not Var or child.function is not Const:
                s += self.count_inner_nodes(child)
        return 1 + s

    def substitute(self, n: Node):
        n_inner = self.count_inner_nodes(n)

        if n_inner == 0:
            return

        p = 1.0 / n_inner
        n_rnd = self.rnd_inner(n, p)
        n_rnd.function = random.choice(self.functions)

    def insert(self, n: Node):
        v, p = self.rnd_leaf(n)
        u = random.choice(self.terminals)
        w = random.choice(self.functions)

        tmp = v.function
        v.function = w
        v.children.append(Node(function=u, children=[]))
        v.children.append(Node(function=tmp, children=[]))

    def delete(self, n: Node):
        v, p = self.rnd_leaf(n)

        if p is None:
            return

        if p.children[0] is v:
            u = p.children[1]
        else:
            u = p.children[0]

        p.function = u.function
        p.children = u.children

    def as_list(self):
        return [self.substitute, self.insert,  self.delete]


class SimpleTGP(TinyTGP):
    hyperparameters: SGPHyperparameters
    def __init__(self, functions_: list, terminals_: list, config_: TGPConfig, hyperparameters_: SGPHyperparameters):
        super().__init__(functions_, terminals_, config_, hyperparameters_)
        self.hvl_prime = HVLPrime(functions_, terminals_).as_list()

    @override
    def init(self):
        self.population = [self.init_individual() for _ in range(self.hyperparameters.lmbda + 1)]

    def init_individual(self) -> TGPIndividual:
        return TGPIndividual(genome_=[self.init_tree_simple()])

    def init_tree_simple(self):
        return Node(function=random.choice(self.terminals), children=[])

    @override
    def perturb(self, parent1: Node, parent2: Node = None) -> Node:
        return self.mutation(parent1)

    @override
    def mutation(self, parent: Node) -> Node:
        for _ in range(self.hyperparameters.k):
            random.choice(self.hvl_prime)(parent)
        return parent

    @override
    def breed(self):
        parent = self.selection()
        self.population = [parent]
        for _ in range(self.hyperparameters.lmbda):
            genome = copy.deepcopy(parent.genome[0])
            offspring = TGPIndividual(
                genome_=[self.perturb(genome)]
            )
            self.population.append(offspring)

    @override
    def selection(self) -> TGPIndividual:
        sorted_pop = sorted(
            self.population,
            key=lambda ind: ind.fitness,
            reverse=not self.config.minimizing_fitness,
        )
        count = 0
        if not self.hyperparameters.strict_selection:
            best_fitness = sorted_pop[0].fitness
            for individual in sorted_pop:
                if individual.fitness != best_fitness:
                    break
                else:
                    count += 1
            parent = random.randint(0, count - 1)
        else:
            parent = 0
        return sorted_pop[parent]

    @override
    def pipeline(self, problem):
        self.breed()
        return self.evaluate(problem)
