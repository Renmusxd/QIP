from qip.qip import *
from qip.operators import *
from qip.util import gen_qubit_prints
from qip.qfft import QFFT, QIFFT
from qip.ext.kronprod import measure_probabilities
import numpy
import math
import random

MAX_QUBITS = 28
MAXN = numpy.ceil(numpy.power(2**(MAX_QUBITS - 2), 0.25))

class Umod(Qubit):
    def __init__(self, *inputs):
        super().__init__(*inputs)


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


def quantum(a, N):
    # Find required number of bits
    q = int(numpy.ceil(numpy.log2(N**2)))
    print(q)
    Q = 2**q

    reg1 = Qubit(n=q)  # Will hold |x>
    reg2 = Qubit(n=q)  # Will hold |f(x)>

    ufreg1, ufreg2 = F(lambda x: (a%N)**x % N, reg1, reg2)

    qft = QFFT(ufreg1)

    # Initial state is superposition of all |x> values.
    reg1_init = numpy.ones((Q,)) * numpy.power(Q,-0.5)
    reg2_init = numpy.zeros((Q,))
    reg2_init[0] = 1.0

    o_qfft, _ = qft.run(feed={reg1: reg1_init, reg2: reg2_init})

    ps = measure_probabilities(numpy.array(list(range(0,q)), dtype=numpy.int32), 2*q, o_qfft)
    for s in gen_qubit_prints(ps, q):
        print(s)

    o, c = Measure(qft).run(feed={reg1: reg1_init, reg2: reg2_init})
    return o, c

o, c = quantum(5,9)

print(c)
print(o)

n = int(numpy.log2(len(o)))

