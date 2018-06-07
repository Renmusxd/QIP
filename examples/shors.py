from qip.qip import *
from qip.operators import *
from qip.util import gen_qubit_prints
from qip.qfft import *
from qip.ext.kronprod import measure_probabilities
from qip.pipeline import *
import numpy
import math
import random
from matplotlib import pyplot


MAX_QUBITS = 28


def classical(N):
    if 2*numpy.log2(N) > MAX_QUBITS:
        raise ValueError("N above MAXN")

    while True:
        a = random.randint(2,N-1)
        print("Trying a={}...".format(a))
        gcd = math.gcd(a, N)
        if gcd != 1:
            print("gcd={}".format(gcd))
            return a
        r = quantum(a, N)
        print("\tQuantum part found r={}...".format(r))
        if r % 2 == 1:
            print("\tR is odd.")
            continue
        apow = a**int(r/2)
        if pow(a, int(r/2), 2) == 1:
            print("\ta^{r/2} % N is odd...")
            continue
        if math.gcd(apow + 1, N) != 1 and math.gcd(apow - 1, N):
            print("Found a={}!".format(a))
            return a


def quantum(x, N):
    n_qubits = int(numpy.ceil(numpy.log2(N)))
    m_qubits = int(numpy.ceil(numpy.log2(x)))

    r = 0
    while r == 0:
        stochastic_output, m = make_circuit(m_qubits, n_qubits,
                                            x, N)
        o, c = run(m)
        r = c[m][0]
        print("\tr={}".format(r))
    return r


def make_circuit(m,n, x, N):
    """
    Make a quantum circuit with m,n qubits
    :param m: qubits in register 1
    :param n: qubits in register 2
    :return: input qubits and output qubits
    """

    # Instead of QFFT just initialize to (1/sqrt(2**m))|x>
    default_state = numpy.ones((2 ** m,)) * pow((2 ** m), -1.0 / 2.0)
    reg1 = Qubit(n=m, default=default_state)  # Superposition of each |i>
    reg2 = Qubit(n=n)  # Will hold |f(i)>, defaults to |0>

    ufreg1, ufreg2 = F(lambda i: pow(x, i, N), reg1, reg2)

    qft = QFFT(ufreg1)

    out1, out2 = Qubit(qft, ufreg2).split()

    # Measure Repeatedly
    mqft = StochasticMeasure(out1)
    m = Measure(qft)

    return mqft, m

classical(21)

#
# mqft, m = make_circuit(9,4,11,21)
#
# printCircuit(mqft)
#
# o, c = run(mqft, m)
# pyplot.plot(c[mqft])
# pyplot.show()
