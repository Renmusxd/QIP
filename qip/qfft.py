from qip.qip import Qubit
from numpy.fft import fft


class QFFT(Qubit):
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
