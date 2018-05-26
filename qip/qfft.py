from qip.qip import Qubit
from qip.ext.qfft import qfft, qifft
from qip.util import flatten
import numpy


class QFFTOp(Qubit):
    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def feed(self, inputvals, qbitindex, n, arena):
        """
        Operate on the state of the system.
        :param inputvals: 2**n complex values
        :param qbitindex: mapping of qbit to global index
        :param n: number of qubits
        :param arena: memory arena to use for computations, must be of size 2**n
        :return: (2**n complex values of applying Q to input, memory arena of size 2**n, (num classic bits, int bits))
        """
        indices = numpy.array(flatten(qbitindex[q] for q in self.inputs), dtype=numpy.int32)
        qfft(indices, n, inputvals, arena)
        return arena, inputvals, (0, 0)


class QIFFTOp(Qubit):
    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def feed(self, inputvals, qbitindex, n, arena):
        """
        Operate on the state of the system.
        :param inputvals: 2**n complex values
        :param qbitindex: mapping of qbit to global index
        :param n: number of qubits
        :param arena: memory arena to use for computations, must be of size 2**n
        :return: (2**n complex values of applying Q to input, memory arena of size 2**n, (num classic bits, int bits))
        """
        indices = numpy.array(flatten(qbitindex[q] for q in self.inputs), dtype=numpy.int32)
        qifft(indices, n, inputvals, arena)
        return arena, inputvals, (0, 0)


def QFFT(*inputs, **kwargs):
    n = QFFTOp(*inputs, **kwargs)
    if len(inputs) > 1:
        return n.split()
    return n


def QIFFT(*inputs, **kwargs):
    n = QIFFTOp(*inputs, **kwargs)
    if len(inputs) > 1:
        return n.split()
    return n