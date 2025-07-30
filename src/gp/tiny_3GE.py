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
        '''
        Triggers the evaluation of the whole population.

        :return: a `float` value of the best fitness
        '''
        best = None
        # For each individual in the population 
        for ix, individual in enumerate(self.population):
            lin_genome = individual.lin_genome   # extract the genome
            fitness = self.evaluate_individual(individual.genome, lin_genome) # evaluate it
            self.population[ix] = TreeGEIndividual(individual.genome, lin_genome, fitness) # assign the fitness
            # update the population best solution
            if best is None or self.problem.is_better(fitness, best):
                best = fitness
            # update the best solution of all time
            if self.best_individual is None or self.problem.is_better(fitness, self.best_individual.fitness):
                self.best_individual = GEIndividual(individual.genome, lin_genome, fitness)
        return best
    
    def evaluate_individual(self, deriv_tree, lin_genome:list[int]) -> float:
        '''
        Evaluate a single individual `genome`.

        :return: a `float` representing the fitness of that individual.
        '''
        self.num_evaluations += 1  # update the evaluation counter
        f = None
        tmp_expr = self.expression(lin_genome)
        if '<' in tmp_expr or '>' in tmp_expr:
            f = self.hyperparameters.penalty_value 
        else:
            f = self.problem.evaluate(lin_genome, self) # evaluate the solution using the problem instance
        if self.best_individual is None or self.problem.is_better(f, self.best_individual.fitness):
            self.best_individual = TreeGEIndividual(deriv_tree, lin_genome, f)
        return f
    
    def predict(self, lin_genome: list, observation: list) -> list:
        '''
        Predict the output of the `genome` given a single `observation`.

        :return: a list of the outputs for that observation
        '''
        def evaluate_expression(expr: str, func_dict: list, args: list[str], values: list) -> any:
            local_vars = dict(zip(args, values))
            return [eval(expr, func_dict, local_vars)]

        tmp_expr = self.expression(lin_genome)    # TODO: expression already generated in evaluate_individual() -> prevent double execution
        return evaluate_expression(tmp_expr, self.functions, self.arguments, observation)
    
    def perturb(self, parent1: TreeGEIndividual, parent2: TreeGEIndividual) -> TreeGEIndividual:
        '''
        Applies the crossover and mutation operators to the parents.

        :return: a list of the `genome` and `None` representing the unevaluated fitness.
        '''
        # applies the crossover with `self.hyperparameters.cx_rate` probability, otherwise return the first parent
        offspring_genome = self.crossover(parent1, parent2) if random.random() <= self.hyperparameters.cx_rate else parent1
        # applies mutation with `self.hyperparameters.mutation_rate`
        offspring_mutated = self.mutation(offspring_genome, self.hyperparameters.mutation_rate)
        return offspring_mutated
    
    def mutation(self, individual: TreeGEIndividual, mutation_rate: float) -> TreeGEIndividual:
        """
        Mutates an individual by replacing a random subtree with a new random subtree.

        :param individual: The individual to mutate.
        :return: A new mutated individual.
        """
        # Deepcopy to avoid in-place mutation

        if random.random() > mutation_rate:  # If mutation is not applied, return the original individual
            return deepcopy(individual)
        
        mutated_tree = deepcopy(individual.genome)
        mutable_nodes = []      # Collect all mutable (non-terminal) nodes with [selected_node, parent, child_index, depth] information

        def collect_nodes(node: Node, parent=None, child_index=None, depth=0):
            if self.is_non_terminal(node.NT) and parent is not None:  # Only consider non-terminal nodes that have a parent 
                mutable_nodes.append((node, parent, child_index, depth))
            for i, child in enumerate(node.children):
                collect_nodes(child, node, i, depth+1)

        collect_nodes(mutated_tree)     # call the function on the root node to collect all mutable nodes

        if not mutable_nodes:
            return deepcopy(individual)

        selected_node, parent, child_index, depth = random.choice(mutable_nodes)    # Randomly select a node to mutate  
        new_subtree = self.init_random_tree_grow(self.hyperparameters.max_depth - depth, selected_node.NT)  # Generate a new subtree using the same symbol (non-terminal)
        parent.children[child_index] = new_subtree

        new_linear_genome = self.generate_linear_genome(mutated_tree, self.hyperparameters.codon_size)
        mutated_individual = TreeGEIndividual(mutated_tree, new_linear_genome)

        return mutated_individual
    

    def crossover(self, parent1: TreeGEIndividual, parent2: TreeGEIndividual) -> TreeGEIndividual:
        """
        Performs one-point crossover between two tree individuals by exchanging subtrees.

        :param parent1: First parent individual.
        :param parent2: Second parent individual.
        :return: A new offspring individual.
        """
        def tree_depth(node: Node) -> int:
                if not node.children:
                    return 1
                return 1 + max(tree_depth(child) for child in node.children)

        for _ in range(10): # try up to 10 times to find a valid crossover point

            offspring1 = deepcopy(parent1.genome)
            donor_tree = deepcopy(parent2.genome)
            
            crossover_nodes = []    # Collect all crossover points (non-terminal nodes with parents) from offspring with [selected_node, parent, child_index, depth] information

            def collect_crossover_nodes(node: Node, parent=None, child_index=None, depth=0):
                if self.is_non_terminal(node.NT) and parent is not None:
                    crossover_nodes.append((node, parent, child_index, depth))
                for i, child in enumerate(node.children):
                    collect_crossover_nodes(child, node, i, depth+1)
            
            collect_crossover_nodes(offspring1)
            
            if not crossover_nodes:
                continue
            
            selected_node, parent, child_index, depth = random.choice(crossover_nodes)  # Select random crossover point in offspring
            selected_symbol = selected_node.NT
            
            # Collect all compatible subtrees from donor (same non-terminal symbol)
            compatible_subtrees = []

            def collect_compatible_subtrees(node: Node, depth):
                if node.NT == selected_symbol and tree_depth(node) <= self.hyperparameters.max_depth - depth:   # check if the non-terminal matches and the subtree depth is compatible
                    compatible_subtrees.append(node)
                for child in node.children:
                    collect_compatible_subtrees(child, depth)
            
            collect_compatible_subtrees(donor_tree,depth)
            
            if not compatible_subtrees:
                continue
            
            donor_subtree = random.choice(compatible_subtrees)  # Select random compatible subtree from donor and replace in offspring
            parent.children[child_index] = deepcopy(donor_subtree)
            new_linear_genome = self.generate_linear_genome(offspring1, self.hyperparameters.codon_size)
            offspring_individual = TreeGEIndividual(offspring1, new_linear_genome)
            
            return offspring_individual
        
        return deepcopy(parent1)
    
    def selection(self) -> list:            
        '''
        Select a parent from the population using the tournament selection method.

        :return: a selected genome.
        '''
        # samples `self.hyperparameters.tournament_size` solutions completely at random
        parents = [random.choice(self.population) for _ in range(self.hyperparameters.tournament_size)]
        # return the best of this sample whether it is a minimization or maximization problem     
        if self.problem.minimizing:
            return min(parents, key=lambda ind: ind.fitness)
        else:
            return max(parents, key=lambda ind: ind.fitness)
        
    def expression(self, lin_genome: list) -> str:
        '''
        Convert a genome into string format with the help of the grammar.

        :return: expression as `str`.
        '''
        return self.genotype_phenotype_mapping(self.grammar, lin_genome, '<expr>')
    
    def genotype_phenotype_mapping(self, grammar, lin_genome, expression='<expr>'):
            '''
                Maps the genotype to its phenotype.
                
                :return: a string representation of the genome.
            '''
            tmp_genome = copy.deepcopy(lin_genome)
            while '<' in expression and len(tmp_genome) > 0:
                next_non_terminal = re.search(r'<(.*?)>', expression).group(0)
                choice = grammar[next_non_terminal][(tmp_genome.pop(0) % len(grammar[next_non_terminal]))]
                expression = expression.replace(next_non_terminal, choice, 1)
            return expression
    
    def breed(self):
        '''
        Breed the population by first selecting a set of pair of parents and then
        applying crossover and mutation operators.
        '''
        # Select n pairs of parents using tournament selection, n is the population size minus 1 (so we have space for the best individual)
        parents = [[self.selection(), self.selection()] for _ in range(self.hyperparameters.pop_size-1)]
        # replace the current population by perturbing the sampled parents     
        self.population = [self.perturb(*parent) for parent in parents]
        # keep the best solution in the population 
        self.population.append(TreeGEIndividual(self.best_individual.genome, self.best_individual.lin_genome, self.best_individual.fitness))
    
    def evolve(self):
        """
        Runs the evolution steps.

        TODO: implement this in the base class as a default for every algorithm.
        """
        # measure the current time     
        t0 = time.time()
        elapsed = 0
        terminate = False
        best_fitness_job = None
        # for each job, if running parallel executions     
        for job in range(self.config.num_jobs):
            best_fitness = None
            # run for a maximum of generations     
            for generation in range(self.config.max_generations):
                # breed     
                self.breed()
                # evaluate the new population
                best_fitness = self.evaluate()
                # `report_generation` will handle the reporting of every generation according to the config     
                self.report_generation(silent_algorithm= self.config.silent_algorithm,
                                       generation=generation,
                                       best_fitness=best_fitness,
                                       report_interval=self.config.report_interval)
                t1 = time.time()
                delta = t1-t0
                t0 = t1
                elapsed += delta
                # if elapsed time is larger than the maximum time, terminate 
                if elapsed + delta >= self.config.max_time:
                    terminate = True
                    break
                # if we found the ideal fitness, terminate         
                elif self.problem.is_ideal(best_fitness):
                    terminate = True
                    break
            # update the current best between runs and report             
            if best_fitness_job is None or self.problem.is_better(best_fitness, best_fitness_job):
                    best_fitness_job = best_fitness
            self.report_job(job = job,
                            num_evaluations=self.num_evaluations,
                            best_fitness=best_fitness_job,
                            silent_evolver=self.config.silent_evolver,
                            minimalistic_output=self.config.minimalistic_output)
            if terminate:
                break
        return self.best_individual.genome
        

    
    """ Initialization methods for derivation trees """


    def init_ramped_half_half(self, min_depth: int, max_depth: int):
        """
        Generates a population of individuals using the Ramped Half and Half method.
        
        :param min_depth: Minimum depth of the derivation tree.
        :param max_depth: Maximum depth of the derivation tree.
        :return: A derivation either generated by Grow or Full.
        """
        depth = random.randrange(min_depth, max_depth+1)  # Randomly choose a depth for the generation method
        method = random.choice(["full", "grow"])      # Randomly choose between full and grow method for tree generation
        if method == "grow":
            return self.init_random_tree_grow(depth, list(self.grammar.keys())[0])
        else:   # use Full method
            return self.init_random_tree_full(depth, list(self.grammar.keys())[0])
        
    
    def init_random_tree_grow(self, max_depth: int, symbol: str):
        """
        Generates a single derivation tree using the random tree method.
        
        :param max_depth: Maximum depth of the derivation tree.
        :param symbol: The symbol to start the derivation tree with (usually a non-terminal).
        :return: A derivation tree generated using the GROW method.
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
        :return: A derivation tree generated using the FULL method.
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
        

    def print_population(self, population: list[TreeGEIndividual]):
        for ind in population:
            self.print_individual_tree(ind.genome, level=1)
            print(f"Linear genome: {ind.lin_genome}\n")

    def print_individual_tree(self, node: Node, level=0):
        indent = "  " * level
        rule_info = f" [rule: {node.production_rule}]" if node.production_rule else ""

        if not self.is_non_terminal(node.NT):  # If the node is a terminal, print it as a leaf
            print(f"{indent}Leaf(Terminal='{node.NT}'){rule_info}")
            return
        print(f"{indent}Node(NT='{node.NT}'){rule_info}")
        for child in node.children:
            if isinstance(child, Node):
                self.print_individual_tree(child, level + 1)
            else:
                print(f"{'  ' * (level + 1)}Leaf(Terminal='{child}')")

    def print_individual(self, individual: TreeGEIndividual):
        """
        Prints information about a single individual.
        """
        expression_str = self.expression(individual.lin_genome)
        print(f"Expression: {' '.join(expression_str)} : Fitness: {individual.fitness}")

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