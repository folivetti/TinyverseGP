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


class Node:

    def __init__(self, symbol, children):
        """
        Initialise an instance of the tree class.
        
        :param expr: A non-terminal from the underlying grammar in BNF.
        :param parent: The parent of the current node. None if node is tree
        root.
        """
        self.NT = symbol
        self.children = children

    # for testing purposes to see the structure of the node (tree) 
    def __repr__(self, level=0):
        indent = "  " * level
        if not self.children:
            # This is a leaf node (terminal)
            return f"{indent}Leaf(NT={self.NT})\n"
        
        repr_str = f"{indent}Node(NT={self.NT})\n"
        for child in self.children:
            if isinstance(child, Node):
                repr_str += child.__repr__(level + 1)
            else:
                repr_str += f"{'  ' * (level + 1)}Leaf(Terminal={child})\n"
        return repr_str

class TreeGEIndividual(GPIndividual):
    genome: list[Node]
    lin_genome: list[int]  # linear representation of the genome (representation format like in tinyGE)
    fitness: any

    def __init__(self, genome: list[Node], lin_genome: list[int], fitness: any = None):
        GPIndividual.__init__(self, genome, fitness)
        lin_genome = lin_genome
        print(self.genome)

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
        self.problem = problem_ # The problem instance to solve, e.g. a symbolic regression problem.
        self.functions = functions_ # List of functions that can be used in the derivation tree (inner nodes).
        self.grammar = grammar_ # Dictionary in BNF format.
        self.arguments = arguments_
        self.config = config
        self.hyperparameters = hyperparameters

        self.root = None
        self.num_evaluations = 0
        self.best_individual = None
        self.best_fitness = None

        self.population = [TreeGEIndividual(deriv_tree, [], 0.0) for deriv_tree in self.init_random_tree_pop(self.hyperparameters.pop_size, 2, list(self.grammar.keys())[0])] # We assume that the first key in the grammar is the start symbol.


    def init_random_tree_pop(self, num_pop: int, max_depth: int, start_symbol: str):

        return [self.init_random_tree(max_depth, start_symbol) for _ in range(num_pop)]
    

    def get_minimum_derivation_steps(self, NT: str, grammar: dict, cache=None, visited=None) -> int:
        """
        Returns the minimum number of derivation steps required to derive a non-terminal NT

        :param NT: The non-terminal symbol to derive.
        :param grammar: The grammar in BNF format.
        :param cache: A dictionary to cache results for previously computed non-terminals.
        :param visited: A set to track visited non-terminals to avoid cycles - especially important to prevent endless recursion.
        """

        if cache is None:
            cache = {}   # use memoization to cache results 
        if visited is None:
            visited = set()
        # Terminal symbol → 0 steps
        if NT not in grammar:   # Check if NT is a key in the grammar dictionary - if not it is a terminal
            return 0
        # If we’ve already computed this, return cached result
        if NT in cache:
            return cache[NT]

        # Detect visited non-terminals to avoid cycles
        if NT in visited:
            return float('inf')  # Avoid cycles, this path is invalid

        visited.add(NT)  # Mark as visited

        min_steps = float('inf')    # Initialize minimum steps to infinity
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
            symbols = self.parse_production(production)
            can_complete = True

            for sym in symbols:
                if self.is_non_terminal(sym):   # Check if the symbol is a non-terminal 
                    min_steps = self.get_minimum_derivation_steps(sym, self.grammar)
                    if min_steps >= max_depth:      # if the minimum steps to derive this symbol is greater than or equal to the remaining depth
                        can_complete = False
                        break
            
            if can_complete:
                valid_productions.append(production)

        return valid_productions
    

    def generate_codon(self, symbol: str) -> int:
        pass

    

    """ Initialization methods for derivation trees """

    
    
    def init_random_tree(self, max_depth: int, symbol: str):
        """
        Generates a single derivation tree using the random tree method.
        
        :param max_depth: Maximum depth of the derivation tree.
        :param symbol: The symbol to start the derivation tree with (usually a non-terminal).
        :return: A single derivation tree (Node).
        """

        cur_NT = symbol     # current non-terminal to derive from   
        possible_productions = self.grammar.get(cur_NT, [])  # Get all possible productions for the current non-terminal
        if max_depth <= 1:      # Check if we've reached maximum depth
            # At maximum depth, return terminal node with empty children
            if not self.is_non_terminal(cur_NT):
                return Node(cur_NT, [])
            
            terminal_productions = [p for p in possible_productions if all(not self.is_non_terminal(s) for s in self.parse_production(p))]  # filter productions to only include those that are terminal because we are already at maximum depth
            # if there are no terminal productions, return None
            # This means that we cannot derive a valid tree from this non-terminal at maximum depth
            # The algorithm retries to build a valid tree
            if not terminal_productions:
                return None
        
        # Check if the current non-terminal has productions
        if not possible_productions:    # check if cur_NT is a terminal (productions are empty)
            # If there are no productions, return a terminal node (leaf node)
            return Node(cur_NT, [])
        
        # Filter productions that can fit within remaining depth
        valid_productions = self.filter_valid_productions(possible_productions, max_depth)

        if not valid_productions: # If no valid productions, return terminal node
            return Node(cur_NT, [])
            
        production = random.choice(valid_productions)   # Randomly select a production from the valid productions
        symbols = self.parse_production(production)     # Parse the production to get individual symbols
        
        # Recursively create child nodes for each symbol in the production
        children = []
        for sym in symbols:
            child = self.init_random_tree(max_depth - 1, sym)    # max_depth - 1 to ensure we don't exceed the maximum depth
            if not child:
                self.init_random_tree(max_depth, cur_NT) 
            else:
                children.append(child)
        return Node(cur_NT, children)        # Create node with the computed children

    
    
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
        return self.population


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

    # abstract
    def evaluate_individual(self,genome:GPIndividual) -> float:
        """
        Fitness function that evaluates a single individual.
        """
        pass
    
    # abstract
    def evolve(self)  -> Any:
        """
        Main evolution loop that is used to run instances
        of a GP model.
        """
        pass

    # abstract
    def selection(self) -> Any:
        """
        Implementation of the selection mechanism.
        Commonly returns an individual object or the position
        of an individual in the population.
        """
        pass

    # abstract    
    def predict(self) -> Any:
        """
        The respective prediction method is implemented here.
        """
        pass

    # abstract 
    def expression(self) -> Any:
        """
        Returns a human-readable solution of a evolved candidate solution.
        Return value can be a string or a list of strings.
        """
        pass

    def is_non_terminal(self, symbol: str) -> bool:
        return symbol.startswith('<') and symbol.endswith('>')  # non-terminals are enclosed in angle brackets

    def parse_production(self, production: str) -> list:
        """
        Parses a production string to extract individual symbols.
        
        :param production: The production string to parse.
        :return: List of symbols in the production.
        example: parse_production("<expr> + <term>") -> ['<expr>', '+', '<term>']
        """
        # This is a basic implementation - you may need to adapt this
        # based on your specific grammar format
        
        # If production is empty or just whitespace, return empty list
        # There are no productions to parse
        if not production or not production.strip():
            return []
        symbols = production.strip().split()    # Simple split by whitespace  
        symbols = [s for s in symbols if s]    # Filter out empty strings

        
        return symbols
    