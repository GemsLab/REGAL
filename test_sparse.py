import numpy as np 
from scipy import sparse
from xnetmf import *
from config import *

A = sparse.csr_matrix( np.random.randint(2,size=(4,4)) )
B = sparse.csr_matrix( np.random.randint(2,size=(4,4)) )
comb = sparse.block_diag([A,B])

graph = Graph(adj = comb.tocsr())
rep_method = RepMethod(max_layer = 2)
representations = get_representations(graph, rep_method)
print representations.shape

