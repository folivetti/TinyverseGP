import random
from typing_extensions import override
from src.gp.tiny_cgp import CGPHyperparameters, CGPConfig, TinyCGP


class SimpleCGP(TinyCGP):

    def __init__(self, functions_: list, terminals_: list, config_: CGPConfig, hyperparameters_: CGPHyperparameters):
        super().__init__(functions_, terminals_, config_, hyperparameters_)
        self.hyperparameters.lmbda = 1

    @override
    def mutation(self, genome: list[int]):
        active_nodes = self.active_nodes(genome)
        active = False
        while not active:
            gene_pos = random.randint(0, self.config.num_genes - 1)
            genome[gene_pos] = self.init_gene(gene_pos)
            node_num = self.node_number(gene_pos)
            if node_num in active_nodes or self.phenotype(gene_pos) == self.GeneType.OUTPUT:
                active = True

