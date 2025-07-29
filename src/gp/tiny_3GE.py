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
- the grammar is defined in BNF format in a dictionary. 
- This module is intended to reuse the existing TinyGE functionalities that are already implemented in the TinyGE class.
- It implements a mapping from the derivation tree to a linear genome representation, which is then used for evaluation and prediction via the TinyGE class.
- This class will use all the hyperparamters defined in the TinyGE class and add only one hyperparater: 'max_depth' for the derivation tree.
"""


import random
from copy import *
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
    """
    max_depth: int
    genome_length: int
    codon_size: int 
    penalty_value: int


class Node:

    def __init__(self, symbol, children, production_rule=None):
        """
        Represents a node in the derivation tree.
        
        :param symbol: The symbol of the node (non-terminal or terminal).
        :param children: List of child nodes.
        :param production_rule: The production rule used to create this node (optional).
        """
        self.NT = symbol
        self.children = children
        self.production_rule = production_rule


class TreeGEIndividual(GPIndividual):
    genome: Node
    lin_genome: list[int]  # linear representation of the genome (representation format like in tinyGE)
    fitness: any

    def __init__(self, genome: list[Node], lin_genome: list[int], fitness: any = None):
        GPIndividual.__init__(self, genome, fitness)
        self.lin_genome = lin_genome


class Tiny3GE(GPModel):

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
        self.functions = {f.name.upper(): f.function for f in functions_} # the list of functions to that could be used in the grammar                                 # TODO: Adjust to updates in the framework
        self.grammar = grammar_     # BNF grammar in dictionary format
        self.arguments = arguments_
        self.config = config
        self.hyperparameters: TreeGEHyperparameters = hyperparameters
        self.num_evaluations = 0
        self.best_individual = None

        self.population: list[TreeGEIndividual] = [TreeGEIndividual(deriv_tree,
                                                self.generate_linear_genome(deriv_tree, self.hyperparameters.codon_size), 0.0) 
                                                for deriv_tree in self.init_random_tree_pop(self.hyperparameters.pop_size, self.hyperparameters.max_depth, list(self.grammar.keys())[0])]     # We assume that the first key in the grammar is the start symbol.

        self.geModel = TinyGE(problem_, functions_, grammar_, arguments_, config, hyperparameters)  # Initialize the TinyGE model to use evolve() method and other functionalities on the linear genomes
        for i in range(self.hyperparameters.pop_size):
            self.geModel.population[i] = GEIndividual(  # linear genome of the derivation tree as genome of GE instance
                self.population[i].lin_genome, 
                self.population[i].fitness
            )

        self.evaluate()
        

    def init_random_tree_pop(self, num_pop: int, max_depth: int, start_symbol: str):
        # return[self.init_random_tree_grow(max_depth, start_symbol) for _ in range(num_pop)]
        # return [self.init_random_tree_full(max_depth, start_symbol) for _ in range(num_pop)]
        return [self.init_ramped_half_half(4, max_depth) for _ in range(num_pop)]
    

    def get_minimum_derivation_steps(self, NT: str, cache=None, visited=None) -> int:
        """
        Returns the minimum number of derivation steps required to derive a non-terminal NT until only terminal symbols are left.

        :param NT: The non-terminal symbol to derive.
        :param cache: A dictionary to cache results for previously computed non-terminals - reference to memoization.
        :param visited: A set to track visited non-terminals to avoid cycles - especially important to prevent endless recursion.
        """

        if cache is None: cache = {}   # use memoization to cache results 
        if visited is None: visited = set()
        if NT not in self.grammar: return 0 # Check if NT is a key in the grammar dictionary - if not it is a terminal
        if NT in cache: return cache[NT]    # If we’ve already computed this, return cached result
        
        if NT in visited: return float('inf')  # Avoid cycles, this path is invalid
        visited.add(NT)  
        min_steps = float('inf')    

        for production in self.grammar[NT]:  # Get all productions for the current non-terminal
            symbols = self.parse_production(production)
            max_child_steps = 0
            for sym in symbols:
                steps = self.get_minimum_derivation_steps(sym, cache, visited.copy())
                max_child_steps = max(max_child_steps, steps)   # maximum steps are required in order to ensure that all children are derived
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
                    min_steps = self.get_minimum_derivation_steps(sym)    # recursively compute minimum steps to derive this symbol
                    if min_steps >= max_depth:  
                        can_complete = False
                        break
            if can_complete:
                valid_productions.append(production)

        return valid_productions
    
    
    def filter_recursive_productions(self, productions: list[str], cur_NT: str) -> list[str]:
        """
        Filters a list of productions to include only those that are recursive.
        A production is considered recursive if the left-hand non-terminal appears in its own right-hand side.
        
        :param productions: List of production strings to evaluate.
        :param cur_NT: The current non-terminal being expanded (i.e. the LHS symbol).
        :return: List of recursive production strings.
        """
        recursive_productions = []
        for production in productions:
            symbols = self.parse_production(production)
            if cur_NT in symbols:
                recursive_productions.append(production)
        return recursive_productions
    

    def generate_codon(self, node: Node, codon_size) -> int:
        """
        Generates a linear representation of the derivation tree (genome) as a list of integers.
        
        :param tree: The derivation tree to convert into a linear representation.
        :return: A list of integers representing the genome.
        """
        # [no. choices, no. choices, codon_size] - [start, step, stop] interval

        num_choices = len(self.grammar[node.NT])
        production_index = self.grammar[node.NT].index(node.production_rule) # Get the index of the production rule in the grammar for the current non-terminal
        offset = random.randrange(0, codon_size - num_choices + 1, num_choices)     # Random offset in num_choices steps to ensure the codon is within the codon size limit

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
            if node.production_rule is not None and self.is_non_terminal(node.NT):      # Add a codon for any node that was generated by expanding a non-terminal
                genome.append(self.generate_codon(node, codon_size))
            for child in node.children:
                build_genome(child)

        build_genome(tree_root)
        
        return genome
    
               
    def evaluate(self) -> float:
        """
        Triggers the evaluation of the whole population.

        :return: a `float` value of the best fitness
        """
        self.geModel.evaluate() # Use the TinyGE model's evaluate method to evaluate the linear genomes
        

    def evolve(self):
        """
        Main evolution loop that is used to run instances
        of a GP model.
        
        :return: the best individual in the population.
        """
        return self.geModel.evolve()
        

    
    """ Initialization methods for derivation trees """

    
    
    def init_random_tree_grow(self, max_depth: int, symbol: str):
        """
        Generates a single derivation tree using the random tree method.
        
        :param max_depth: Maximum depth of the derivation tree.
        :param symbol: The symbol to start the derivation tree with (usually a non-terminal).
        :return: A single derivation tree (Node).
        """
        cur_NT = symbol      
        possible_productions = self.grammar.get(cur_NT, [])  

        if not possible_productions:    # check if cur_NT is a terminal (productions are empty)
            return Node(cur_NT, [], None)
        
        valid_productions = self.filter_valid_productions(possible_productions, max_depth)      # filter all the productions to those that can be derived within the remaining depth 
            
        production = random.choice(valid_productions)
        symbols = self.parse_production(production)     
        
        # Recursively create child nodes for each symbol in the production
        children = []
        for sym in symbols:
            child = self.init_random_tree_grow(max_depth - 1, sym) 
            children.append(child)

        return Node(cur_NT, children, production)    
    

    def init_random_tree_full(self, remaining_depth: int, symbol: str) -> Node:
        """
        Generates a derivation tree using the FULL method, where all branches extend to exactly max_depth.

        :param remaining_depth: The remaining depth allowed for the derivation tree.
        :param symbol: The current symbol to expand.
        """
        cur_NT = symbol
        possible_productions = self.grammar.get(cur_NT, [])

        # If we're at the last depth level, pick only terminal productions
        if remaining_depth <= self.get_minimum_derivation_steps(cur_NT):
            if not possible_productions:    # check if cur_NT is a terminal (productions are empty)
                return Node(cur_NT, [], None)
            
            valid_productions = self.filter_valid_productions(possible_productions, remaining_depth)      # filter all the productions to those that can be derived within the remaining depth    
            production = random.choice(valid_productions)
            symbols = self.parse_production(production)     
            
            # Recursively create child nodes for each symbol in the production
            children = []
            for sym in symbols:
                child = self.init_random_tree_grow(remaining_depth - 1, sym)  
                children.append(child)
            return Node(cur_NT, children, production) 
                    
        if not possible_productions:  # check if cur_NT is a terminal (productions are empty)
            return Node(cur_NT, [], None)
        
        recursive_productions = self.filter_recursive_productions(possible_productions, cur_NT)     # We're not at max_depth: choose only recursive productions
        production = random.choice(recursive_productions)
        symbols = self.parse_production(production)

        children = []
        for sym in symbols:
            child = self.init_random_tree_full(remaining_depth - 1, sym)
            children.append(child)

        return Node(cur_NT, children, production)


    def init_ramped_half_half(self, min_depth: int, max_depth: int):
        """
        Generates a population of individuals using the Ramped Half and Half method.
        
        :param min_depth: Minimum depth of the derivation tree.
        :param max_depth: Maximum depth of the derivation tree.
        :return: A list of individuals (derivation trees).
        """
        depth = random.randrange(min_depth, max_depth+1)  # Randomly choose a depth for the generation method
        method = random.choice(["full", "grow"])      # Randomly choose between full and grow method for tree generation
        if method == "grow":
            return self.init_random_tree_grow(depth, list(self.grammar.keys())[0])
        else:   # use Full method
            return self.init_random_tree_full(depth, list(self.grammar.keys())[0])
        
        

    def print_population(self, population: list[TreeGEIndividual]):
        for ind in population:
            self.print_individual(ind.genome, level=1)
            print(f"Linear genome: {ind.lin_genome}\n")

    
    def print_individual(self, node: Node, level=0):
        indent = "  " * level
        rule_info = f" [rule: {node.production_rule}]" if node.production_rule else ""

        if not self.is_non_terminal(node.NT):  # If the node is a terminal, print it as a leaf
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
            Parses a production rule into its components (terminals and non-terminals).

            :param production: A string representing a production rule in the grammar.
            :return: A list of symbols (terminals and non-terminals) parsed from the production rule.
            """
            if not production or not production.strip():
                return []
            # pattern = re.compile(r'<[^<> ]+>|[A-Za-z_]+|[(),.]')
            pattern = re.compile(r'<[^<> ]+>|\d+|[A-Za-z_]+|[(),.]')
            return pattern.findall(production)        
           
    
    # abstract
    def selection(self) -> Any:
        """
        Implementation of the selection mechanism.
        Commonly returns an individual object or the position
        of an individual in the population.
        """
        pass
    # @abstractmethod
    def evaluate_individual(self,genome:GPIndividual) -> float:
        """
        Fitness function that evaluates a single individual.
        """
        pass
    # @abstractmethod
    def selection(self) -> Any:
        """
        Implementation of the selection mechanism.
        Commonly returns an individual object or the position
        of an individual in the population.
        """
        pass
    # @abstractmethod
    def predict(self) -> Any:
        """
        The respective prediction method is implemented here.
        """
        pass
    # @abstractmethod
    def expression(self) -> Any:
        """
        Returns a human-readable solution of a evolved candidate solution.
        Return value can be a string or a list of strings.
        """
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

    # 
    
    def mutation(self, individual: TreeGEIndividual) -> TreeGEIndividual:
        """
        Mutates an individual by replacing a random subtree with a new random subtree.

        :param individual: The individual to mutate.
        :return: A new mutated individual.
        """
        # Deepcopy to avoid in-place mutation
        mutated_tree = deepcopy(individual.genome)
        depth = 0
        # Collect all mutable (non-terminal) nodes with their parents and child indices
        mutable_nodes = []

        def collect_nodes(node: Node, parent=None, child_index=None, depth=0):
            if self.is_non_terminal(node.NT) and parent is not None:  # Only consider non-terminal nodes that have a parent 
                mutable_nodes.append((node, parent, child_index, depth))
            for i, child in enumerate(node.children):
                collect_nodes(child, node, i, depth+1)

        collect_nodes(mutated_tree)     # call the function on the root node to collect all mutable nodes

        # If no mutable nodes, return original individual
        if not mutable_nodes:
            return deepcopy(individual)

        # Randomly select a node to mutate
        selected_node, parent, child_index, depth = random.choice(mutable_nodes)   
        print("selected_node: ---------------------------------------------------------------------------------------------------------------------")
        self.print_individual(selected_node, level=1)
        print("parent: ---------------------------------------------------------------------------------------------------------------------")
        self.print_individual(parent, level=1)
        print(child_index)
        print("depth:", depth)

        # Generate a new subtree using the same symbol (non-terminal)
        # max_depth = self.hyperparameters.max_depth
        new_subtree = self.init_random_tree_grow(self.hyperparameters.max_depth - depth, selected_node.NT)

        # Replace the selected subtree in the parent
        if parent is None:
            # Mutating the root node
            mutated_tree = new_subtree
        else:
            parent.children[child_index] = new_subtree

        # Rebuild linear genome
        new_linear_genome = self.generate_linear_genome(mutated_tree, self.hyperparameters.codon_size)

        # Create new individual
        mutated_individual = TreeGEIndividual(mutated_tree, new_linear_genome)

        return mutated_individual
    