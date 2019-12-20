from qip.operators import *
from qip.svgwriter import make_svg


q1 = Qubit(n=1)
q2 = Qubit(n=3)
q3 = Qubit(n=3)

h1 = H(q1)
c1, c2, c3 = C(Swap)(h1, q2, q3)
m1 = H(c1)

# Make example circuit diagram
make_svg(m1, filename='cswap.svg')

# CSwap compare should give P(|q1=0>) = 1/2 + 1/2<q2|q3> = 1/2
state2 = numpy.cos(numpy.arange(0,2**q2.n) * numpy.pi/(2.0**q2.n))
state3 = numpy.sin(numpy.arange(0,2**q3.n) * numpy.pi/(2.0**q3.n))

state2 = state2 / numpy.linalg.norm(state2)
state3 = state3 / numpy.linalg.norm(state3)

o, _ = m1.run(feed={q1: [1.0, 0.0], q2: state2, q3: state3})