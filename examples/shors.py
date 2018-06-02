from qip.qip import *
from qip.operators import *
from qip.util import gen_qubit_prints
from qip.qfft import QFFT
from qip.ext.kronprod import measure_probabilities
from qip.pipeline import run
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

    mufreg1 = StochasticMeasure(ufreg1)

    qft = QFFT(mufreg1)

    # Measure Repeatedly
    mqft = StochasticMeasure(qft)
    mreg1 = StochasticMeasure(ufreg2)

    return mqft, mufreg1, mreg1

def quantum(a, N):
    # Find required number of bits
    q = int(numpy.ceil(numpy.log2(N**2)))
    print(q)
    Q = 2**q

    # Initial state is superposition of all |x> values.
    reg1_init = numpy.ones((Q,)) * numpy.power(Q,-0.5)

    reg1 = Qubit(n=q, default=reg1_init)  # Will hold |x>
    reg2 = Qubit(n=q)  # Will hold |f(x)>

    ufreg1, ufreg2 = F(lambda x: pow(a, x, N), reg1, reg2)

    qft = QFFT(ufreg1)

    o_qfft, _ = qft.run()

    ps = measure_probabilities(numpy.array(list(range(0,q)), dtype=numpy.int32), 2*q, o_qfft)
    for s in gen_qubit_prints(ps, q):
        print(s)

    o, c = Measure(qft).run()
    return o, c


mqft, mufrag1, mreg1 = make_circuit(8,4,11,15)
o, c = run(mqft, mreg1)
print(c[mqft])
pyplot.plot(c[mreg1])
pyplot.show()
pyplot.plot(c[mufrag1])
pyplot.show()
pyplot.plot(c[mqft])
pyplot.show()
# o, c = quantum(5,9)
#
# print(c)
# print(o)
#
# n = int(numpy.log2(len(o)))
#
