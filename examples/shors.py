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
MAXN = numpy.ceil(numpy.power(2**(MAX_QUBITS - 2), 0.25))


def classical(N):
    if N > MAXN:
        raise ValueError("N above MAXN")

    while True:
        a = random.randint(2,N)
        print("Trying a={}...".format(a))
        gcd = math.gcd(a, N)
        if gcd != 1:
            return a
        r = quantum(a, N)
        print("\tQuantum part found r={}...".format(r))
        if r % 2 == 1:
            print("\tR is odd.")
            continue
        apow = a**(r/2)
        if apow % N == 1:
            print("\ta^{r/2} % N is odd...")
            continue
        if math.gcd(apow + 1, N) != 1 and math.gcd(apow - 1, N):
            print("Found a={}!".format(a))
            return a


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
    mreg2 = StochasticMeasure(out2)

    return mqft, mreg2


mqft, mreg2 = make_circuit(9,4,11,21)

printCircuit(mqft, mreg2)


o, c = run(mqft, mreg2)
# print(c[mreg2])
# print(c[mqft])
pyplot.plot(c[mreg2])
pyplot.show()
pyplot.plot(c[mqft])
pyplot.show()
