import random
from dataclasses import dataclass
from typing_extensions import override
from src.gp.tiny_cgp import CGPHyperparameters, CGPConfig, TinyCGP

@dataclass(kw_only=True)
class SimpleCGPConfig(CGPConfig):
    log_scaling: bool = False

class SimpleCGP(TinyCGP):

    def __init__(self, functions_: list, terminals_: list, config_: SimpleCGPConfig, hyperparameters_: CGPHyperparameters):
        super().__init__(functions_, terminals_, config_, hyperparameters_)
        self.hyperparameters.lmbda = 1

    def new_value(self, position: int, old_val: int=None) -> int:
        gene_type = self.phenotype(position)

        if old_val is None:
            exclude = []
        else:
            exclude = [old_val]

        if gene_type == self.GeneType.CONNECTION:
            levels_back = self.hyperparameters.levels_back
            node_num = self.node_number(position)
            if node_num <= levels_back:
                min = 0
            else:
                min = node_num - levels_back
            max = node_num - 1
        elif gene_type == self.GeneType.FUNCTION:
            min = 0
            max = self.config.num_functions - 1
        else:
            min = 0
            max = self.config.num_inputs + self.hyperparameters.num_function_nodes - 1

        new_vals = [c for c in range(min, max + 1) if c not in exclude]
        return random.choice(new_vals) if len(new_vals) > 0 else old_val

    @override
    def mutation(self, genome: list[int]):
        active_nodes = self.active_nodes(genome)
        active = False
        while not active:
            gene_pos = random.randint(0, self.config.num_genes - 1)
            gene_val = genome[gene_pos]
            genome[gene_pos] = self.new_value(gene_pos, old_val=gene_val)
            node_num = self.node_number(gene_pos)
            if node_num in active_nodes or self.phenotype(gene_pos) == self.GeneType.OUTPUT:
                active = True

