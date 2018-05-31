from qip.qip import Qubit
from qip.operators import H, C, R
from qip.ext.qfft import qfft, qifft
from qip.util import flatten
import numpy


def QFFT(*inputs, **kwargs):
    """
    Implementation of Quantum Fourier Transform using H and Rphi gates.
    :return:
    """
    makeR = lambda phi: (lambda *inputs, **kwargs: R(phi, *inputs, **kwargs))

    # Make into array of qubits of n=1
    qarr = flatten([inp.split(range(inp.n)) for inp in inputs])
    for i in range(len(inputs)):
        # Apply rotations
        for j in range(i - 1):
            phi = numpy.pi / (2 ** (i - j))
            qarr[i], qarr[j] = C(makeR(phi))(qarr[i], qarr[j], **kwargs)
        # Apply Hadamard
        qarr[i] = H(qarr[i], **kwargs)

    # Regroup by original qubits
    outputqarr = []
    index = 0
    for qubit in inputs:
        outputqarr.append(Qubit(*(qarr[index:index+qubit.n])))
        index += qubit.n

    if len(outputqarr) > 1:
        return tuple(outputqarr)
    else:
        return outputqarr[0]


# Experimental code for using numpy's qfft library.
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


# FFTs using the specialized op instead of successive H + C(rotation)s.
# May have uses, though recursive version is more real.
def pyQFFT(*inputs, **kwargs):
    n = QFFTOp(*inputs, **kwargs)
    if len(inputs) > 1:
        return n.split()
    return n


def pyQIFFT(*inputs, **kwargs):
    n = QIFFTOp(*inputs, **kwargs)
    if len(inputs) > 1:
        return n.split()
    return n