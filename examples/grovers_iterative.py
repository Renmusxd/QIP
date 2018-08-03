from qip.operators import *
from qip.qip import *
from qip.qfft import *
from qip.svgwriter import make_svg

try:
    from matplotlib import pyplot
except ImportError:
    pyplot = False


def graph(measure):
    if pyplot:
        pyplot.plot(measure, label='ds')
        pyplot.ylim([0, 1])
        pyplot.legend()
        pyplot.show()


def Uw(search, ancillary, x0):
    return F(lambda x: int(x == x0), search, ancillary)


def Us(search, ancillary):
    f_s, f_anc = Uw(H(search), ancillary, 0)
    return H(f_s), f_anc


# Number of qubits and "criteria"
n = 10
x = 42

# Set up both groups in superposition states
q = Qubit(n=n, default=numpy.ones(shape=(2 ** n,)) / numpy.sqrt(2 ** n))
anc = Qubit(n=1, default=[1 / numpy.sqrt(2), -1 / numpy.sqrt(2)])

oracle_search, oracle_anc = Uw(q, anc, x)
diff_search, diff_anc = Us(oracle_search, oracle_anc)
sm_ds, sm_da = StochasticMeasure(diff_search), StochasticMeasure(diff_anc)

# Make example circuit diagram
make_svg(sm_ds, filename='grovers.svg')

state, c = run(sm_ds, sm_da)

graph(c[sm_ds])

# At each iteration of grovers feed the state from the last iteration, as though it were one long circuit.
for i in range(int(2 ** (n / 2)) - 1):
    state, c = run(sm_ds, sm_da, state=state)
    graph(c[sm_ds])

print(numpy.argmax(c[sm_ds]))