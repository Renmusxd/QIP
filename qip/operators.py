from qip.qip import Qubit
from qip.util import kronselect_dot
from qip.util import flatten
import numpy


class MatrixOp(Qubit):
    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)
        self.ms = None

    def feed(self, inputvals, qbitindex, n, arena):
        """
        Operate on the state of the system.
        :param inputvals: 2**n complex values
        :param qbitindex: mapping of qbit to global index
        :param n: number of qubits
        :param arena: memory arena to use for computations, must be of size 2**n
        :return: (2**n complex values of applying Q to input, memory arena of size 2**n, (num classic bits, int bits))
        """
        if self.ms is None:
            self.ms = self.makemats(qbitindex)

        kronselect_dot(self.ms, inputvals, n, arena)
        return arena, inputvals,  (0, 0)

    def makemats(self, qbitindex):
        raise NotImplemented("This method should never be called.")


def Not(*inputs, **kwargs):
    n = NotOp(*inputs, **kwargs)
    if len(inputs) > 1:
        return n.split()
    return n


class NotOp(MatrixOp):
    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def makemats(self, qbitindex):
        return {i: numpy.flip(numpy.eye(2), 0)
                for i in flatten([qbitindex[inp] for inp in self.inputs])}

    def __repr__(self):
        return "Not({})".format(self.qid)


def H(*inputs, **kwargs):
    h = HOp(*inputs, **kwargs)
    if len(inputs) > 1:
        return h.split()
    return h


class HOp(MatrixOp):
    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def makemats(self, qbitindex):
        return {i: (1/numpy.sqrt(2))*numpy.array([[1, 1], [1, -1]])
                for i in flatten([qbitindex[inp] for inp in self.inputs])}

    def __repr__(self):
        return "H({})".format(self.qid)


def Swap(*inputs, **kwargs):
    s = SwapOp(*inputs, **kwargs)
    if len(inputs) > 1:
        return s.split()
    return s


class SwapOp(MatrixOp):
    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)
        if len(self.inputs) != 2:
            raise Exception("Swap can only take two inputs")
        if self.inputs[0].n != self.inputs[1].n:
            raise Exception("Inputs must be of equal size {}/{}".format(self.inputs[0], self.inputs[1]))

    def makemats(self, qbitindex):
        swapn = self.inputs[0].n
        a_indices = qbitindex[self.inputs[0]]
        b_indices = qbitindex[self.inputs[1]]
        return {tuple(flatten([a_indices, b_indices])): SwapMat(swapn)}

    def __repr__(self):
        return "Swap({})".format(self.qid)


class SwapMat(object):
    _kron_struct = 3

    def __init__(self, n):
        """
        Constructs a 2^(2n) x 2^(2n) matrix to swap positions of blocks of n entries.
        :param n: size of swap
        """
        self.n = n
        self.shape = (2**(2*n), 2**(2*n))

    def __getitem__(self, item):
        if type(item) == tuple and len(item) == 2:
            low_a, high_a = item[0] & ~(-1 << self.n), item[0] >> self.n
            low_b, high_b = item[1] & ~(-1 << self.n), item[1] >> self.n
            return 1.0 if low_a == high_b and low_b == high_a else 0.0

        else:
            raise ValueError("SwapMat can only be indexed with M[i,j] not M[{}]".format(item))

    def __repr__(self):
        return "SwapMat({})".format(self.n)


def C(op):
    """
    Constructs the controlled version of a given qubit operation
    :param op: operation to control
    :return: A Class C-Op which takes as a first input the controlling qubit and
    remaining inputs as a normal op.
    """
    return lambda *inputs: COp(op, *inputs).split()


class COp(MatrixOp):
    def __init__(self, op, *inputs, **kwargs):
        if len(inputs) < 2:
            raise ValueError("Not enough input values given.")
        if inputs[0].n != 1:
            raise ValueError("Control bit can only be of size n=1")
        self.op = op(*inputs[1:], nosink=True, **kwargs)
        super().__init__(*inputs, qid=self.op.qid, **kwargs)

    def makemats(self, qbitindex):
        opm = self.op.makemats(qbitindex)
        newdict = {}
        for indices in opm:
            newindices = tuple(flatten([qbitindex[self.inputs[0]], indices]))
            newdict[newindices] = CMat(opm[indices])
        return newdict

    def __repr__(self):
        return "C{}".format(self.op)


class CMat(object):
    _kron_struct = 2

    def __init__(self, mat):
        if type(mat) == list:
            self.m = numpy.array(mat)
        else:
            self.m = mat
        self.shape = (self.m.shape[0]*2, self.m.shape[1]*2)

    def __getitem__(self, item):
        if type(item) == tuple and len(item) == 2:
            row, col = item[0], item[1]
            if row < self.shape[0]/2 and col < self.shape[1]/2:
                return 1.0 if row == col else 0.0
            elif row >= self.shape[0]/2 and col >= self.shape[1]/2:
                r, c = row - int(self.shape[0]/2), col - int(self.shape[1]/2)
                return self.m[r, c]
            else:
                return 0.0
        else:
            raise ValueError("CMat can only be indexed with M[i,j] not M[{}]".format(item))
