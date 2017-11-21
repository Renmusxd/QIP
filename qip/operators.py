from qip.qip import Qubit
from qip.util import kronselect_dot
from qip.util import flatten
import numpy


class MatrixOp(Qubit):
    def __init__(self, *inputs):
        super().__init__(*inputs)
        self.ms = None

    def feed(self, inputvals, qbitindex, n):
        """
        Operate on the state of the system.
        :param inputvals: 2**n complex values
        :param qbitindex: mapping of qbit to global index
        :return: 2**n complex values
        """
        if self.ms is None:
            self.ms = self.makemats(qbitindex)
        return kronselect_dot(self.ms, inputvals, n)

    def makemats(self, qbitindex):
        # Identity
        # return {i: numpy.eye(2)
        #         for i in flatten([qbitindex[inp] for inp in self.inputs])}
        raise NotImplemented("This method should never be called.")

class Not(MatrixOp):
    def __init__(self, *inputs):
        super().__init__(*inputs)

    def makemats(self, qbitindex):
        return {i: numpy.flip(numpy.eye(2), 0)
                for i in flatten([qbitindex[inp] for inp in self.inputs])}

    def __repr__(self):
        return "Not({})".format(self._qid)


class H(MatrixOp):
    def __init__(self, *inputs):
        super().__init__(*inputs)

    def makemats(self, qbitindex):
        return {i: (1/numpy.sqrt(2))*numpy.flip(numpy.array([[1,1],[1,-1]]), 0)
                for i in flatten([qbitindex[inp] for inp in self.inputs])}

    def __repr__(self):
        return "H({})".format(self._qid)


class Swap(MatrixOp):
    def __init__(self, *inputs):
        super().__init__(*inputs)
        if len(self.inputs) != 2:
            raise Exception("Swap can only take two inputs")
        if self.inputs[0].n != self.inputs[1].n:
            raise Exception("Inputs must be of equal size {}/{}".format(self.inputs[0],self.inputs[1]))

    def makemats(self, qbitindex):
        swapn = self.inputs[0].n
        a_indices = qbitindex[self.inputs[0]]
        b_indices = qbitindex[self.inputs[1]]
        return {tuple(flatten([a_indices,b_indices])): SwapMat(swapn)}


    def __repr__(self):
        return "Swap({})".format(self._qid)


class SwapMat(object):
    def __init__(self, n):
        """
        Constructs a 2^(2n) x 2^(2n) matrix to swap positions of blocks of n entries.
        :param n: size of swap
        """
        self.n = n
        self.shape = (2**(2*n), 2**(2*n))

    def __getitem__(self, item):
        if type(item) == tuple and len(item) == 2:
            low_a, high_a = item[0] % (2**self.n), item[0] >> self.n
            low_b, high_b = item[1] % (2**self.n), item[1] >> self.n
            return 1.0 if low_a==high_b and low_b==high_a else 0.0

        else:
            raise ValueError("SwapMat can only be indexed with M[i,j] not M[{}]".format(item))

    def __repr__(self):
        return "SwapMat({})".format(self.n)