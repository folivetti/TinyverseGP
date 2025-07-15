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

        self.population = [] # List of individuals in the population - derivation tree representation (Node as elements).

    def is_non_terminal(symbol: str) -> bool:
        return symbol.startswith('<') and symbol.endswith('>')
    

    def get_minimum_derivation_steps(self, NT, grammar: dict) -> int:
        """
        Returns the minimum number of derivation steps required to derive a non-terminal NT
        """
        # Check if NT is in grammar (is a non-terminal)
        if NT not in grammar:
            # If not in grammar, assume it's a terminal - 0 steps needed
            return 0
            
        min_steps = float('inf')    # Initialize to infinity 
        possible_productions = self.grammar[symbol]   # Get all possible productions for the non-terminal NT that remain in the depth

        for production in possible_productions:
            # Parse the production to get symbols
            symbols = self.parse_production(production)

            if not symbols:   # If the production is empty, it means this production is terminal
                # This production is terminal → 1 step
                min_steps = min(min_steps, 1)
            else:
                # Recursively compute derivation steps for each non-terminal in this production
                max_child_steps = 0
                for symbol in symbols:
                    child_steps = self.get_minimum_derivation_steps(symbol, grammar)
                    max_child_steps = max(max_child_steps, child_steps)     # Find the maximum steps needed among all children
                steps = 1 + max_child_steps
                min_steps = min(min_steps, steps)

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
                min_steps = self.get_minimum_derivation_steps(sym, self.grammar)
                if min_steps >= max_depth:      # if the minimum steps to derive this symbol is greater than or equal to the remaining depth
                    can_complete = False
                    break
            
            if can_complete:
                valid_productions.append(production)

        return valid_productions

    

    """ Initialization methods for derivation trees """

    

    def init_random_tree(self, max_depth: int, symbol: str):
        """
        Generates a single derivation tree using the random tree method.
        
        :param max_depth: Maximum depth of the derivation tree.
        :param symbol: The symbol to start the derivation tree with (usually a non-terminal).
        :return: A single derivation tree (Node).
        """

        cur_NT = symbol     # current non-terminal to derive from
        
        # Check if we've reached maximum depth
        if max_depth <= 1:
            # At maximum depth, return terminal node with empty children
            return Node(cur_NT, [])


        # cur_NT = '<expr>'
        # grammar.get(cur_NT, [])  # → ['<expr> + <term>', '<term>']    
        possible_productions = self.grammar.get(cur_NT, [])  # Get all possible productions for the current non-terminal
        


        # Check if the current non-terminal has productions
        if not possible_productions:    # check if cur_NT is a terminal (productions are empty)
            # If there are no productions, return a terminal node (leaf node)
            return Node(cur_NT, [])
        
        # Filter productions that can fit within remaining depth
        valid_productions = self.filter_valid_productions(possible_productions, max_depth)
        
        # If no valid productions, return terminal node
        if not valid_productions:
            return Node(cur_NT, [])
            
        production = random.choice(valid_productions)   # Randomly select a production from the valid productions

        # Parse the production to get individual symbols
        symbols = self.parse_production(production)
        
        # Recursively create child nodes for each symbol in the production
        children = []
        for symbol in symbols:
            child = self.init_random_tree(max_depth - 1, symbol)    # max_depth - 1 to ensure we don't exceed the maximum depth
            children.append(child)

        # Create node with the computed children
        return Node(cur_NT, children)
        


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

    
    """ Initialization methods for derivation trees"""


    def is_valid(self, genome:GPIndividual) -> bool:
        """
        Checks if the genome is valid.
        """
        pass


    def eval_complexity(self, genome:GPIndividual) -> float:
        """
        Evaluates the complexity of the genome.
        """
        pass


    def evaluate_individual(self,genome:GPIndividual) -> float:
        """
        Fitness function that evaluates a single individual.
        """
        pass
    

    def evolve(self)  -> Any:
        """
        Main evolution loop that is used to run instances
        of a GP model.
        """
        pass


    def selection(self) -> Any:
        """
        Implementation of the selection mechanism.
        Commonly returns an individual object or the position
        of an individual in the population.
        """
        pass


    def predict(self) -> Any:
        """
        The respective prediction method is implemented here.
        """
        pass


    def expression(self) -> Any:
        """
        Returns a human-readable solution of a evolved candidate solution.
        Return value can be a string or a list of strings.
        """
        pass

    def parse_production(self, production: str) -> list:
        """
        Parses a production string to extract individual symbols.
        
        :param production: The production string to parse.
        :return: List of symbols in the production.
        """
        # This is a basic implementation - you may need to adapt this
        # based on your specific grammar format
        
        # If production is empty or just whitespace, return empty list
        if not production or not production.strip():
            return []
        
        # Simple split by whitespace - adjust based on your grammar format
        # You might need more sophisticated parsing for complex grammars
        symbols = production.strip().split()
        
        # Filter out empty strings
        symbols = [s for s in symbols if s]
        
        return symbols


