"""
Tiny3GE: A minimalistic implementation of derivation tree based Grammatical Evolution (GE) for tinyverseGP.
This module extends the TinyGE class to support a derivation tree representation of individuals.

        Genome representation: derivation tree
"""

"""
notes:
- This implementation is designed to be minimal and focused on the derivation tree structure.
- define a parameter for the maximum depth of the derivation tree.
    - however, this is optional and can be adjusted based on the problem requirements.
"""


import random
import copy
import time
import re
from src.gp.problem import *
from src.gp.tiny_ge import *
from src.gp.tinyverse import *


@dataclass
class TreeGEHyperparameters(GPHyperparameters):
    """
    Hyperparameters for the Tiny3GE model.
    
    :param max_depth: Maximum depth of the derivation tree.
    :genome_length: Length of the genome (number of codons).
    :codon_size: Size of each codon in the genome.
    :penalty_value: Penalty value for invalid individuals.
    """

    max_depth: int
    genome_length: int
    codon_size: int 
    penalty_value: float # Default penalty value 


class Node:

    def __init__(self, symbol, children, production_rule=None):
        """
        Initialise an instance of the tree class.
        
        :param expr: A non-terminal from the underlying grammar in BNF.
        :param parent: The parent of the current node. None if node is tree
        root.
        """
        self.NT = symbol
        self.children = children
        self.production_rule = production_rule


class TreeGEIndividual(GPIndividual):
    genome: Node
    lin_genome: list[int]
    fitness: any

    def __init__(self, genome: Node, lin_genome: list[int], fitness: any = None):
        self.genome = genome
        self.lin_genome = lin_genome
        self.fitness = fitness

class Tiny3GE1(GPModel):

    '''
    Main class of the tiny3GE module that derives from GPModel and
    implements all related fundamental mechanisms to run GE.
    '''
    config: Config
    hyperparameters: Hyperparameters
    problem: Problem
    functions: list[Function]

    def __init__(self, problem_: object, functions_: list[Function], grammar_: dict, arguments_: list[str], config: Config, hyperparameters: Hyperparameters, ):
        self.problem = problem_ 
        self.functions = {f.name.upper(): f.function for f in functions_} # the list of functions to that could be used in the grammar
        self.grammar = grammar_     # BNF grammar in dictionary format
        self.arguments = arguments_
        self.config = config
        self.hyperparameters = hyperparameters

        self.root = None
        self.num_evaluations = 0
        self.best_individual = None
        self.best_fitness = 0

        self.population = [TreeGEIndividual(deriv_tree,
                                             self.generate_linear_genome(deriv_tree, self.hyperparameters.codon_size), 0.0) 
                           for deriv_tree in self.init_random_tree_pop(self.hyperparameters.pop_size, self.hyperparameters.max_depth, list(self.grammar.keys())[0])]     # We assume that the first key in the grammar is the start symbol.
        self.evaluate()
        self.print_population(self.population)
        
        # Initialize the TinyGE model with the population by using the linear genomes
        self.geModel = TinyGE(problem_, functions_, grammar_, arguments_, config, hyperparameters)  # Initialize the TinyGE model
        for i in range(self.hyperparameters.pop_size):
            self.geModel.population[i] = GEIndividual(  # mutable danger prevention
                self.population[i].lin_genome, 
                self.population[i].fitness
            )

        
    def init_random_tree_pop(self, num_pop: int, max_depth: int, start_symbol: str):

        return [self.init_random_tree(max_depth, start_symbol) for _ in range(num_pop)]
    

    def get_minimum_derivation_steps(self, NT: str, grammar: dict, cache=None, visited=None) -> int:
        """
        Returns the minimum number of derivation steps required to derive a non-terminal NT until only terminal symbols are left.

        :param NT: The non-terminal symbol to derive.
        :param grammar: The grammar in BNF format.
        :param cache: A dictionary to cache results for previously computed non-terminals - reference to memoization.
        :param visited: A set to track visited non-terminals to avoid cycles - especially important to prevent endless recursion.
        """

        if cache is None: cache = {}   # use memoization to cache results 
        if visited is None: visited = set()
        if NT not in grammar: return 0 # Check if NT is a key in the grammar dictionary - if not it is a terminal
        if NT in cache: return cache[NT]    # If we’ve already computed this, return cached result
        
        if NT in visited: return float('inf')  # Avoid cycles, this path is invalid
        visited.add(NT)  
        min_steps = float('inf')    

        for production in grammar[NT]:  # Get all productions for the current non-terminal
            symbols = self.parse_production(production)
            max_child_steps = 0
            for sym in symbols:
                steps = self.get_minimum_derivation_steps(sym, grammar, cache, visited.copy())
                max_child_steps = max(max_child_steps, steps)
            total_steps = 1 + max_child_steps
            min_steps = min(min_steps, total_steps)
        cache[NT] = min_steps    # Cache the result for the current non-terminal
        return min_steps 
    

    def filter_valid_productions(self, productions: list[str], max_depth: int) -> list[str]:
        """
        Filters a list of productions to include only those that can be completed
        within the given maximum depth of the derivation tree.

        :param productions: List of production strings to evaluate.
        :param max_depth: Remaining depth allowed in the derivation tree.
        :return: A filtered list of productions that can be fully expanded within the given depth.
        """
        valid_productions = []

        for production in productions:
            symbols = self.parse_production(production)     # extract individual symbols from the production
            can_complete = True     # flag to check if the production can be completed within the remaining depth
            for sym in symbols:
                if self.is_non_terminal(sym): 
                    min_steps = self.get_minimum_derivation_steps(sym, self.grammar)    # recursively compute minimum steps to derive this symbol
                    if min_steps >= max_depth:  
                        can_complete = False
                        break
            if can_complete:
                valid_productions.append(production)

        return valid_productions


    def generate_codon(self, node: Node, codon_size) -> int:
        """
        Generates a linear representation of the derivation tree (genome) as a list of integers.
        
        :param tree: The derivation tree to convert into a linear representation.
        :return: A list of integers representing the genome.
        """
        # [no. choices, no. choices, codon_size] - [start, step, stop]

        num_choices = len(self.grammar[node.NT])
        production_index = self.grammar[node.NT].index(node.production_rule) # Get the index of the production rule in the grammar for the current non-terminal
        offset = random.randrange(0, codon_size - num_choices + 1, num_choices) 

        return offset+production_index
    

    def generate_linear_genome(self, tree_root: Node, codon_size: int) -> list[int]:
        """
        Recursively generates a linear genome from a derivation tree.

        :param root: The root node of the derivation tree
        :param codon_size: Max codon value
        :return: List of integers representing the genome
        """
        genome = []
        # Recursive function to traverse the tree and generate the genome
        def build_genome(node: Node):
            if node.children:  # Only generate codons for non-terminal expansions
                genome.append(self.generate_codon(node, codon_size))
            for child in node.children:
                build_genome(child)

        build_genome(tree_root)
        
        # Ensure minimum genome length for TinyGE compatibility
        # TinyGE crossover requires at least 2 codons
        # while len(genome) < 2:
        #     genome.append(random.randint(0, codon_size - 1))
            
        return genome
    

    def evaluate(self) -> float:
        '''
        Triggers the evaluation of the whole population.

        :return: a `float` value of the best fitness
        '''
        best = None
        # For each individual in the population 
        for ix, individual in enumerate(self.population):
            genome = individual.genome   # extract the genome (tree-derivation)
            fitness = self.evaluate_individual(individual) # evaluate it
            #print(fitness)
            self.population[ix] = TreeGEIndividual(genome, individual.lin_genome, fitness) # assign the fitness
            # update the population best solution
            if best is None or self.problem.is_better(fitness, best):
                best = fitness
            # update the best solution of all time
            if self.best_individual is None or self.problem.is_better(fitness, self.best_individual.fitness):
                self.best_individual = TreeGEIndividual(individual.genome, individual.lin_genome, fitness)
        return best
    

    def evaluate_individual(self, genome: TreeGEIndividual) -> float:
        '''
        Evaluate a single individual `genome`.

        :return: a `float` representing the fitness of that individual.
        '''
        self.num_evaluations += 1  # update the evaluation counter
        f = None
        tmp_expr = self.expression(genome.lin_genome)  # map the genome to its phenotype
        if '<' in tmp_expr or '>' in tmp_expr:
            f = self.hyperparameters.penalty_value 
        else:
            f = self.problem.evaluate(genome.lin_genome, self) # evaluate the solution using the problem instance
        if self.best_individual is None or self.problem.is_better(f, self.best_individual.fitness):
            self.best_individual = TreeGEIndividual(genome.genome, genome.lin_genome, f)
            self.best_fitness = f
        return f
    
    def expression(self, genome: list) -> str:
        '''
        Convert a genome into string format with the help of the grammar.

        :return: expression as `str`.
        '''
        return self.genotype_phenotype_mapping(self.grammar, genome, '<expr>')
    

    def genotype_phenotype_mapping(self, grammar, genome, expression='<expr>'):
        '''
        Maps the genotype to its phenotype.
        
        :return: a string representation of the genome.
        '''
        tmp_genome = copy.deepcopy(genome)
        # print("genome here_____________________________________________________________", genome)
        while '<' in expression and len(tmp_genome) > 0:
            next_non_terminal = re.search(r'<(.*?)>', expression).group(0)
            choice = grammar[next_non_terminal][(tmp_genome.pop(0) % len(grammar[next_non_terminal]))]
            expression = expression.replace(next_non_terminal, choice, 1)
        # print("here--------------------------------------------------------------------", expression)
        return expression
    
    def predict(self, genome: list, observation: list) -> list:
        '''
        Predict the output of the `genome` given a single `observation`.

        :return: a list of the outputs for that observation
        '''
        def evaluate_expression(expr: str, func_dict: list, args: list[str], values: list) -> any:
            
            local_vars = dict(zip(args, values))
            return [eval(expr, func_dict, local_vars)]

        tmp_expr = self.expression(genome)    # TODO: expression already generated in evaluate_individual() -> prevent double execution
        return evaluate_expression(tmp_expr, self.functions, self.arguments, observation)

    
    """ Initialization methods for derivation trees """

    
    
    def init_random_tree(self, max_depth: int, symbol: str):
        """
        Generates a single derivation tree using the random tree method.
        
        :param max_depth: Maximum depth of the derivation tree.
        :param symbol: The symbol to start the derivation tree with (usually a non-terminal).
        :return: A single derivation tree (Node).
        """
        cur_NT = symbol      
        possible_productions = self.grammar.get(cur_NT, [])  
        if max_depth <= 1:      # Check if maximum depth is reached
            if not self.is_non_terminal(cur_NT):
                return Node(cur_NT, []) # If we are at maximum depth and the current symbol is a terminal, return a leaf node
            terminal_productions = [p for p in possible_productions if all(not self.is_non_terminal(s) for s in self.parse_production(p))]  # filter productions to only include those that are terminal because we are already at maximum depth
            if not terminal_productions:
                return None     # return None if there are no terminal productions available at this depth
            
        if not possible_productions:    # check if cur_NT is a terminal (productions are empty)
            return Node(cur_NT, [])
        
        valid_productions = self.filter_valid_productions(possible_productions, max_depth) 

        if not valid_productions: # If no valid productions, return terminal node
            return Node(cur_NT, [])
            
        production = random.choice(valid_productions)
        symbols = self.parse_production(production)     
        
        # Recursively create child nodes for each symbol in the production
        children = []
        for sym in symbols:
            child = self.init_random_tree(max_depth - 1, sym)  
            while child is None:    # init_random_tree(...) returns None if it cannot derive a valid tree from the current non-terminal
                child = self.init_random_tree(max_depth, cur_NT) # Retry with the current non-terminal if child is None
            children.append(child)

        return Node(cur_NT, children, production)    


    def init_ramped_half_half(self, num_pop: int, min_depth: int, max_depth: int, max_size: int):
        """
        Generates a population of individuals using the Ramped Half and Half method.
        
        :num_pop: Number of individuals in the population.
        :param max_depth: Maximum depth of the derivation tree.
        :param min_depth: Minimum depth of the derivation tree.
        :return: A list of individuals (derivation trees).
        """
        pass


    def generate_population(self, num_pop: int, max_depth: int, start_symbol: str):
        """
        Generate a population of derivation trees.
        
        :param num_pop: Number of individuals in the population.
        :param max_depth: Maximum depth for trees.
        :param start_symbol: The start symbol for the grammar.
        :return: List of tree individuals.
        """
        self.population = []
        for _ in range(num_pop):
            tree = self.init_random_tree(max_depth, start_symbol)
            self.population.append(tree)

        for element in self.population:
            self.generate_linear_genome(element, self.hyperparameters.codon_size)  # Generate linear genome for each individual in the population
        return self.population


    def evolve(self):
        pass


    def selection(self) -> list:
        pass


    # abstract
    def is_valid(self, genome:GPIndividual) -> bool:
        """
        Checks if the genome is valid.
        """
        pass

    # abstract
    def eval_complexity(self, genome:GPIndividual) -> float:
        """
        Evaluates the complexity of the genome.
        """
        pass
    

    def print_population(self, population: list[TreeGEIndividual]):
        for ind in population:
            self.print_individual(ind.genome, level=1)
            print(f"Linear genome: {ind.lin_genome}\n")

    
    def print_individual(self, node: Node, level=0):
        indent = "  " * level
        rule_info = f" [rule: {node.production_rule}]" if node.production_rule else ""
        
        if not node.children:
            print(f"{indent}Leaf(Terminal='{node.NT}'){rule_info}")
            return

        print(f"{indent}Node(NT='{node.NT}'){rule_info}")
        for child in node.children:
            if isinstance(child, Node):
                self.print_individual(child, level + 1)
            else:
                print(f"{'  ' * (level + 1)}Leaf(Terminal='{child}')")


    def is_non_terminal(self, symbol: str) -> bool:
        return symbol.startswith('<') and symbol.endswith('>')  # non-terminals are enclosed in angle brackets


    def parse_production(self, production: str) -> list:
        """
        Parses a production string to extract symbols, including terminals and non-terminals.

        Example: parse_production("<expr>+<term>") -> ['<expr>', '+', '<term>']
        
        """
        if not production or not production.strip():
            return []
        # if not production or not production.strip():
        #     return []
        # Regex: match non-terminals like <...> or individual characters
        pattern = re.compile(r'<[^<> ]+>|[^\s]')
        return pattern.findall(production)
    

if __name__ == "__main__":
    grammar = {
    '<expr>': [
        'ADD(<expr>, <expr>)', 'SUB(<expr>, <expr>)', 'MUL(<expr>, <expr>)', 'DIV(<expr>, <expr>)', 
        '<d>', '<d>.<d><d>', '<var>'
    ],
    '<d>': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0'],
    '<var>': ['x']
    }
    print(list(grammar.keys()))


# def generate_derivation_tree(self, lin_genome: list[int]) -> Node:
    #     """
    #     Generates a derivation tree from a linear genome.

    #     :param lin_genome: The linear genome to convert into a derivation tree.
    #     :return: A Node representing the root of the derivation tree.
    #     """
    #     genome_iter = iter(lin_genome)

    #     def build_tree(symbol: str) -> Node:
    #         if not self.is_non_terminal(symbol):
    #             return Node(symbol, [], None)
    #         productions = self.grammar[symbol]
    #         try:
    #             codon = next(genome_iter)
    #         except StopIteration:
    #             # If genome runs out, fallback to first production
    #             production_index = 0
    #         else:
    #             production_index = codon % len(productions)
    #         production = productions[production_index]
    #         symbols = self.parse_production(production)
    #         children = [build_tree(sym) for sym in symbols]
    #         return Node(symbol, children, production)

    #     start_symbol = list(self.grammar.keys())[0]
    #     return build_tree(start_symbol)


    def crossover(self, parent1: TreeGEIndividual, parent2: TreeGEIndividual) -> TreeGEIndividual:
        """
        Performs crossover between two parent individuals to create a new individual.
        
        :param parent1: The first parent individual.
        :param parent2: The second parent individual.
        :return: A new individual created by crossover.
        """
        # Randomly choose a subtree from each parent
        subtree1 = self.get_random_subtree(parent1.genome)
        subtree2 = self.get_random_subtree(parent2.genome)

        # Create a new genome by replacing the chosen subtrees
        new_genome = deepcopy(parent1.genome)
        self.replace_subtree(new_genome, subtree1, subtree2)

        # Generate the linear genome representation
        lin_genome = self.generate_linear_genome(new_genome, self.hyperparameters.codon_size)

        return TreeGEIndividual(new_genome, lin_genome)
    
    def get_random_subtree(self, node: Node) -> Node:
        """
        Selects a random subtree from the given node.
        
        :param node: The root node of the tree from which to select a subtree.
        :return: A randomly selected subtree (Node).
        """
        if not node.children:
            return node
        # Randomly select a child node
        child = random.choice(node.children)
        # Recursively get a subtree from the selected child
        return self.get_random_subtree(child)
    

    def replace_subtree(self, parent: Node, old_subtree: Node, new_subtree: Node):
        """
        Replaces a subtree in the parent node with a new subtree.
        
        :param parent: The parent node in which to replace the subtree.
        :param old_subtree: The subtree to be replaced.
        :param new_subtree: The new subtree to insert.
        """
        if parent == old_subtree:
            # Replace the entire parent with the new subtree
            return new_subtree
        

    def generate_derivation_tree(self, lin_genome: list[int]) -> Node:
        """
        Generates a derivation tree from a linear genome.

        :param lin_genome: The linear genome to convert into a derivation tree.
        :return: A Node representing the root of the derivation tree.
        """
        genome = lin_genome
        codon_index = 0
        genome_len = len(genome)

        def build_tree(symbol: str) -> Node:
            nonlocal codon_index    # Use nonlocal to modify codon_index in the inner function
            if not self.is_non_terminal(symbol):
                return Node(symbol, [], None)
            productions = self.grammar[symbol]
            codon = genome[codon_index % genome_len]    # Get the codon from the genome, wrapping around if necessary
            codon_index += 1
            production_index = codon % len(productions)
            production = productions[production_index]
            symbols = self.parse_production(production)
            children = [build_tree(sym) for sym in symbols]
            return Node(symbol, children, production)

        return build_tree(list(self.grammar.keys())[0])
    

    # def crossover(self, parent1: TreeGEIndividual, parent2: TreeGEIndividual) -> TreeGEIndividual:
    #     """
    #     Performs crossover between two parent individuals to create a new individual.
        
    #     :param parent1: The first parent individual.
    #     :param parent2: The second parent individual.
    #     :return: A new individual created by crossover.
    #     """

    #     def tree_depth(node: Node) -> int:
    #         if not node.children:
    #             return 1
    #         return 1 + max(tree_depth(child) for child in node.children)

    #     for _ in range(10):
    #         nt_nodes1 = self.get_non_terminal_nodes(parent1.genome)  # Randomly select a non-terminal from the grammar
    #         nt_nodes2 = self.get_non_terminal_nodes(parent2.genome)

    #         if not nt_nodes1 or not nt_nodes2:
    #             continue
            
    #         # selected_node1 = random.choice(nt_nodes1)
    #         # sym = selected_node1.NT
    #         selected_node1 = nt_nodes1
    #         sym = nt_nodes1
    #         print(sym + "\n")
            
    #         # matching_ndoes2 = [n for n in nt_nodes2 if n.NT == sym]
    #         # if not matching_ndoes2:
    #         #     continue

    #         # selected2 = random.choice(matching_ndoes2)

    #         selected2 = nt_nodes2

    #         new_tree = deepcopy(parent1.genome)
    #         sucess = self.replace_subtree(new_tree, selected_node1, deepcopy(selected2))  # Replace the selected subtree in the new tree with the selected subtree from parent2

    #         if tree_depth(new_tree) <= self.hyperparameters.max_depth and sucess:
    #             new_lin_genome = self.generate_linear_genome(new_tree, self.hyperparameters.codon_size)
    #             return TreeGEIndividual(new_tree, new_lin_genome)
    #         # Replace the selected subtree in the new tree with the selected subtree from parent2

    #     return deepcopy(parent1)  # If no valid crossover was found, return a copy of parent1
    
    # def get_non_terminal_nodes(self, node: Node) -> list[Node]:
    #         nodes = []
    #         def collect(n):
    #             if self.is_non_terminal(n.NT):
    #                 nodes.append(n)
    #             for c in n.children:
    #                 collect(c)
    #         collect(node)
    #         return nodes
    
    # def replace_subtree(self, parent: Node, target: Node, replacement: Node) -> bool:
    #     """
    #     Replaces the subtree rooted at `target` in `parent` with `replacement`.
    #     Returns True if replacement was successful.
    #     """
    #     if parent == target:
    #         # Can't replace root directly from here — handled externally if needed
    #         return False

    #     for i, child in enumerate(parent.children):
    #         if child == target:
    #             parent.children[i] = replacement
    #             return True
    #         elif self.replace_subtree(child, target, replacement):
    #             return True

    #     return False
    
    # def mutation(self, individual: TreeGEIndividual) -> TreeGEIndividual:
    #     """
    #     Mutates an individual by replacing a random subtree with a new random subtree.
        
    #     :param individual: The individual to mutate.
    #     :return: A new mutated individual.
    #     """


    #     def count_nodes(node: Node) -> int:
    #         return 1 + sum(count_nodes(child) for child in node.children)
        
    #     new_tree = deepcopy(individual.genome)
    #     mutation_num = random.choice(range(count_nodes(individual.genome)))

    #     for i in range(mutation_num):

        
    #     if not non_terminal_nodes:
    #         return deepcopy(individual)
    
    # def get_random_subtree(self, node: Node) -> Node:
    #     """
    #     Selects a random subtree from the given node.
        
    #     :param node: The root node of the tree from which to select a subtree.
    #     :return: A randomly selected subtree (Node).
    #     """
    #     if not node.children:
    #         return node
    #     # Randomly select a child node
    #     child = random.choice(node.children)    # does not work correctly
    #     # Recursively get a subtree from the selected child
    #     return self.get_random_subtree(child)



    # def get_random_subtree(self, node: Node) -> Node:
    #     """
    #     Selects a random subtree from the given node.
        
    #     :param node: The root node of the tree from which to select a subtree.
    #     :return: A randomly selected subtree (Node).
    #     """
    #     if not node.children:
    #         return node
    #     # Randomly select a child node
    #     child = random.choice(node.children)    # does not work correctly
    #     # Recursively get a subtree from the selected child
    #     return self.get_random_subtree(child)
    
    
    # def crossover(self, parent1: TreeGEIndividual, parent2: TreeGEIndividual) -> TreeGEIndividual:
    #     """
    #     Performs crossover between two parent individuals to create a new individual.
        
    #     :param parent1: The first parent individual.
    #     :param parent2: The second parent individual.
    #     :return: A new individual created by crossover.
    #     """

    #     def tree_depth(node: Node) -> int:
    #         if not node.children:
    #             return 1
    #         return 1 + max(tree_depth(child) for child in node.children)

    #     for _ in range(10):
    #         nt_nodes1 = self.get_non_terminal_nodes(parent1.genome)  # Randomly select a non-terminal from the grammar
    #         nt_nodes2 = self.get_non_terminal_nodes(parent2.genome)

    #         if not nt_nodes1 or not nt_nodes2:
    #             continue
            
    #         # selected_node1 = random.choice(nt_nodes1)
    #         # sym = selected_node1.NT
    #         selected_node1 = nt_nodes1
    #         sym = nt_nodes1
    #         print(sym + "\n")
            
    #         # matching_ndoes2 = [n for n in nt_nodes2 if n.NT == sym]
    #         # if not matching_ndoes2:
    #         #     continue

    #         # selected2 = random.choice(matching_ndoes2)

    #         selected2 = nt_nodes2

    #         new_tree = deepcopy(parent1.genome)
    #         sucess = self.replace_subtree(new_tree, selected_node1, deepcopy(selected2))  # Replace the selected subtree in the new tree with the selected subtree from parent2

    #         if tree_depth(new_tree) <= self.hyperparameters.max_depth and sucess:
    #             new_lin_genome = self.generate_linear_genome(new_tree, self.hyperparameters.codon_size)
    #             return TreeGEIndividual(new_tree, new_lin_genome)
    #         # Replace the selected subtree in the new tree with the selected subtree from parent2

    #     return deepcopy(parent1)  # If no valid crossover was found, return a copy of parent1
    
    # def get_non_terminal_nodes(self, node: Node) -> list[Node]:
    #         nodes = []
    #         def collect(n):
    #             if self.is_non_terminal(n.NT):
    #                 nodes.append(n)
    #             for c in n.children:
    #                 collect(c)
    #         collect(node)
    #         return nodes
    
    # def replace_subtree(self, parent: Node, target: Node, replacement: Node) -> bool:
    #     """
    #     Replaces the subtree rooted at `target` in `parent` with `replacement`.
    #     Returns True if replacement was successful.
    #     """
    #     if parent == target:
    #         # Can't replace root directly from here — handled externally if needed
    #         return False

    #     for i, child in enumerate(parent.children):
    #         if child == target:
    #             parent.children[i] = replacement
    #             return True
    #         elif self.replace_subtree(child, target, replacement):
    #             return True

    #     return False