from qip.qip import *
from qip.operators import *
from qip.qfft import *
from qip.pipeline import *
from qip.svgwriter import make_svg
import numpy
import math
import random

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
            return gcd, int(N/gcd)
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
        if factors[0] == 1 or factors[1] == 1:
            continue
        return factors


def quantum(x, N):
    q_qubits = int(numpy.ceil(numpy.log2(N**2)))
    q = pow(2, q_qubits)
    n_qubits = int(numpy.ceil(numpy.log2(N)))
    # m_qubits = int(numpy.ceil(numpy.log2(x)))

    max_iters = 10
    r = 1
    while r == 1 and max_iters > 0:
        stochastic_output, m = make_circuit(q_qubits, n_qubits, x, N)
        o, c = run(m)
        peak = c[m][0]
        if peak == 0:
            continue

        # (peak / q) ~= (d/r)
        print("\tmeasured peak: {}".format(peak))
        for d, r in reversed(list(gen_continued_fractions(peak, q))):
            print("\tcontinued fractions: {},{}".format(d,r))
            # Coprimes between d/r = 0 and 1
            if 1 < d < r < q and r%2 == 0 and math.gcd(d,r) == 1:
                break

        print("\tx={} \tr={}".format(x,r))
        max_iters -= 1
    return r


def gen_continued_fractions(c, q):
    """
    Yields continued fraction approximations for c/q
    """
    coefs = gen_continued_fraction_coefs(c,q)

    dn_2 = 0
    dn_1 = 1
    rn_2 = 1
    rn_1 = 0
    for an in coefs:
        dn = an * dn_1 + dn_2
        rn = an * rn_1 + rn_2
        yield dn, rn
        dn_2, dn_1 = dn_1, dn
        rn_2, rn_1 = rn_1, rn


def gen_continued_fraction_coefs(c, q):
    """
    Yields an for all n such that
    c/q = a0 + 1/(a1 + 1/(...))
    """
    while c > 0 and q > 0:
        a = int(c/q)
        c, q = q, c - q*a
        yield a


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


# Make an example circuit diagram
mqft, m = make_circuit(9,4,11,21)
make_svg(mqft, filename='shors.svg')

# Run shors on 21=7*3
factors = classical(21)
print("Factors: {}".format(factors))
