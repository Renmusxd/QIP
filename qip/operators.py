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
        return {i: numpy.eye(2)
                for i in flatten([qbitindex[inp] for inp in self.inputs])}


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

    def makemats(self, qbitindex):
        pass

    def __repr__(self):
        return "Swap({})".format(self._qid)