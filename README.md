# This is a cloned version of TinyverseGP that has been anonymized.

- It serves to compare Tree-based GP (TGP) and Cartesian GP (CGP) on the well-known MAX benchmark
- Provides two simplified GP models that have been used in literature to perform runtime analysis
  - `src/analysis/models/tiny_tgp.py`: implementation of (1+1)-TGP 
  - `src/analysis/models/tiny_cgp.py`: implementation of (1+1)-CGP
- The MAX problem is provided in both forms Max-depth-D-{+}-{t} (MaxPlus) and Max-depth-D-{+,*}-{t} (MaxPlusMul)
  - `src/analysis/problems.py`: implementation of MaxPlus and MaxPlusMul
  

