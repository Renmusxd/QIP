from qip.qip import Qubit
import numpy
from scipy.linalg import block_diag
from scipy.sparse import lil_matrix
from scipy.sparse import identity


class MatrixOp(Qubit):
    def __init__(self, *inputs):
        super().__init__(*inputs)
        self.m = None

    def feed(self, inputvals, savem=True):
        if self.m is None:
            m = self.makemat(self.n)
            if savem:
                self.m = m
        else:
            m = self.m
        return m.dot(inputvals)

    def makemat(self, nqbits):
        return identity(2**nqbits)


class Not(Qubit):
    def __init__(self, *inputs):
        super().__init__(*inputs)

    def feed(self, inputvals):
        return numpy.flip(inputvals, 0)


class H(MatrixOp):
    def __init__(self, *inputs):
        super().__init__(*inputs)

    def makemat(self, nqbits):
        pass


class Swap(MatrixOp):
    def __init__(self, *inputs):
        super().__init__(*inputs)

    def makemat(self, nqbits):
        pass