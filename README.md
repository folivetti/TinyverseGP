# This branch of TinyverseGP that has been anonymized and freezed for peer-review

- It serves as a codebase for recombination-based Cartesian Genetic Programming (CGP)
- Implementations for two recombination operators are provided:
    - Discrete Phenotypic Recombination
    - Subgraph Crossover
- The operators have been implemented in TinyCGP:
  - [src/gp/tiny_cgp.py](https://github.com/GPBench/TinyverseGP/blob/recombination-based-cgp/src/gp/tiny_cgp.py)
  - [subgraph_crossover](https://github.com/GPBench/TinyverseGP/blob/1573e8eb387f17343d31a0d8cb580bfa8cd9f59d/src/gp/tiny_cgp.py#L670)
  - [discrete_recombination](https://github.com/GPBench/TinyverseGP/blob/1573e8eb387f17343d31a0d8cb580bfa8cd9f59d/src/gp/tiny_cgp.py#L737)
- Breeding pipleline used to run recombination-based CGP:
    - [breed](https://github.com/GPBench/TinyverseGP/blob/234257782fb4d8fb61ad89a15446a9c671935b87/src/gp/tiny_cgp.py#L568)         
