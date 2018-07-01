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
        print("Step 1: pick a < N={}".format(N))
        a = random.randint(2,N-1)
        print("\ta={}...".format(a))

        print("Step 2: compute gcd(a,N)")
        gcd = math.gcd(a, N)
        print("\tgcd({},{})={}".format(a,N,gcd))
        if gcd != 1:
            print("gcd != 1 --> Factor found!")
            return a
        print("Step 3: Find period r")
        r = quantum(a, N)
        print("\tr={}".format(r))
        if r % 2 == 1:
            print("\tr is odd, returning to step 1.")
            continue
        apow = pow(a,int(r/2), N)
        if apow == N - 1:
            print("\ta^{r/2} % N === -1, returning to step 1.")
            continue

        factors = (math.gcd(apow+1,N), math.gcd(apow-1,N))
        return factors[0]


def quantum(x, N):
    n_qubits = int(numpy.ceil(numpy.log2(N)))
    m_qubits = int(numpy.ceil(numpy.log2(x)))

    r = 1
    while r == 1:
        stochastic_output, m = make_circuit(m_qubits, n_qubits,
                                            x, N)
        o, c = run(m)
        x = c[m][0]
        if x==0:
            continue

        print("\tx={}".format(x))
        r = math.gcd(x,N)

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
#
# mqft, m = make_circuit(9,4,11,21)
#
# printCircuit(mqft)
#
# o, c = run(mqft, m)
# pyplot.plot(c[mqft])
# pyplot.show()
#
# top_ten = list(reversed(c[mqft].argsort()[-10:]))
#
# print(top_ten)
# print(c[mqft][top_ten])