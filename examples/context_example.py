from qip.qip import *
from qip.operators import *
from qip.svgwriter import *
from qip.util import QubitFuncWrapper


@QubitFuncWrapper.wrap
def basic_circuit(qa, qb):
    h = H(qa)
    CNot = C(Not)
    return CNot(h, qb)

def main():
    a = Qubit(n=1)
    b = Qubit(n=1)
    c = Qubit(n=1)

    cout, aout, bout = C(basic_circuit)(c, a, b)

    make_svg(cout, 'context.svg')

main()