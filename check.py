from kronprod import cdot_loop
import numpy
from qip.util import dot_loop, expand_kron_matrix
from qip.operators import SwapMat, CMat

indexgroups = numpy.array([ numpy.array([0,1,2]) ])
#matrices = [SwapMat(1)]
matrices = [ CMat(SwapMat(1)) ]

n = 3
vec = numpy.arange(0,n,1)
vec = vec / float(sum(vec))

out = numpy.zeros(shape=vec.shape)

matdict = {tuple(indexgroups[i]): matrices[i] for i in range(len(indexgroups))}


# cdot_loop(indexgroups, matrices, vec, 2, out)
#
# print(out)
#
# print("="*30)
#
# dot_loop(matdict, vec, 2, out)
#
# print(out)

print(expand_kron_matrix(matdict, n, cmode=True))