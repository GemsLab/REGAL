This repository contains a Python implementation of methods presented in the following paper:

Mark Heimann, Haoming Shen, Tara Safavi, and Danai Koutra. *REGAL: Representation Learning-based Graph Alignment*. International Conference on Information and Knowledge Management (CIKM), 2018.

*Paper*: https://gemslab.github.io/papers/heimann-2018-regal.pdf

Please consider citing this paper if you find the code helpful. 
```bibtex
@inproceedings{DBLP:conf/cikm/HeimannSSK18,
  author    = {Mark Heimann and
               Haoming Shen and
               Tara Safavi and
               Danai Koutra},
  title     = {{REGAL:} Representation Learning-based Graph Alignment},
  booktitle = {Proceedings of the 27th {ACM} International Conference on Information
               and Knowledge Management, {CIKM} 2018, Torino, Italy, October 22-26,
               2018},
  pages     = {117--126},
  publisher = {{ACM}},
  year      = {2018},
}
```

Included is code for REGAL, our node embedding framework for network alignment, and its component node embedding method xNetMF.   
This is only a reference implementation; without doubt it can be much improved, but we hope it is helpful!


DEPENDENCIES
=======================
numpy, scipy, networkx (all may be installed with pip)
Tested with Python 3.8.5 and Python 2.7.16

EXAMPLE
=======================
- Align (unattributed): python regal.py 
- Align (with attributes): python regal.py --attributes data/attributes/attr1-2vals-prob0.000000
- Embedding only: python xnetmf.py
- (example runs on data/arenas990-1 dataset, one of the trials of the Arenas email network with 1% noise)

INSTRUCTIONS - xNetMF embeddings only
=======================
- Import config (see config for details about the options on the following objects)
- Initialize a Graph object with adjacency matrix and any other optional information (e.g. node attributes)
- Initialize a RepMethod object, with whatever hyperparameter settings you wish to use (defaults are preset)
- Call get_representations() in xnetmf.py with these two objects as arguments

INSTRUCTIONS - REGAL alignments from xNetMF embeddings
======================
- Combine two graphs with adjacency matrices A1, A2, into combined matrix: [A1 0; 0 A2]
- Save this graph as an edgelist file (recommended: NetworkX write_edgelist() method) as DATA_combined_edges.txt (the "combined_edges.txt" part is optional, but regal.py will look for what is before the first underscore)
- Save a dictionary of the true alignments {node in graph 1 : counterpart in graph 2} as a pickle in DATA_edges-mapping-permutation.txt in the same folder (regal.py will look for this file)
  (note: the formulation in the paper allows graphs of different sizes to be aligned, but this code is written for graphs of the same size.  It should be possible to modify the scoring in alignments.py to handle this, if desired)
- Pass the path to the DATA_combined_edges.txt file as a command line argument to regal.py with the --input flag
- Specify a file to save embeddings to with an --output flag
- OPTIONAL (for REGAL or xNetMF): save attributes for each node as (n_nodes x n_attributes) .npy matrix
- See parse_args() regal.py for more details
