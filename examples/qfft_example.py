from qip.qip import *
from qip.qfft import *
from qip.pipeline import *
from qip.svgwriter import make_svg
import numpy

n = 6
freqs = [3, 5.5]
q = Qubit(n=n)
mq = StochasticMeasure(q)
qft = QFFT(mq, rev=True)
mqft = StochasticMeasure(qft)

make_svg(mqft, filename='qfft.svg')

states = [numpy.cos(numpy.linspace(0,2*freq*numpy.pi,2**n)) for freq in freqs]

state = numpy.zeros(2**n, dtype=numpy.complex128)
for s in states:
    state = state + s

state = state / numpy.linalg.norm(state)
o, c = run(mqft, feed={q: state})

print(c[mqft])

# If plotting tools installed.
try:
    from matplotlib import pyplot
    pyplot.plot(c[mq])
    pyplot.show()

    pyplot.plot(c[mqft])
    pyplot.show()
except:
    pass
