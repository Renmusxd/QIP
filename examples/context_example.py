from qip.qip import *
from qip.operators import *
from qip.svgwriter import *
from qip.qubit_util import QubitFuncWrapper


@QubitFuncWrapper.wrap
def basic_circuit(qa, qb):
    h = H(qa)
    CNot = C(Not)
    return CNot(h, qb)

def main():
    c = Qubit(n=1)
    a = Qubit(n=1)
    b = Qubit(n=1)

    CNot = C(C(Not))
    # c, a, b = (c, H(a), b)

    cout, aout, bout = C(basic_circuit)(c, a, b)

    make_svg(cout, filename='context.svg')

main()