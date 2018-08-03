import unittest
from qip.qip import *
from qip.operators import *
from qip.qubit_util import *
from qip.pipeline import run


class TestQubitMethods(unittest.TestCase):

    def test_basic_context(self):
        q1 = Qubit(n=1, default=[pow(2,-0.5), pow(2, -0.5)])
        q2 = Qubit(n=1)

        # Normal circuit
        c1, c2 = C(H)(q1, q2)
        o1, _ = run(c1, c2)

        # Context circuit
        q3 = Qubit(n=1, default=[pow(2, -0.5), pow(2, -0.5)])
        q4 = Qubit(n=1)

        with QubitWrapperContext(C, [q3]):
            h = H(q4)

        o2, _ = run(h)

        self.assertTrue(numpy.allclose(o1, o2))

    def test_explicit_context(self):
        q1 = Qubit(n=1, default=[pow(2, -0.5), pow(2, -0.5)])
        q2 = Qubit(n=1, default=[pow(2, -0.5), pow(2, -0.5)])
        q3 = Qubit(n=1)

        # Normal circuit
        c1, c2, c3 = C(C(H))(q1, q2, q3)
        o1, _ = run(c1, c2, c3)

        # Context circuit
        q4 = Qubit(n=1, default=[pow(2, -0.5), pow(2, -0.5)])
        q5 = Qubit(n=1, default=[pow(2, -0.5), pow(2, -0.5)])
        q6 = Qubit(n=1)

        with QubitWrapperContext(C, [q4]):
            c4, c5 = C(H)(q5, q6)
        o2, _ = run(c4, c5)

        self.assertTrue(numpy.allclose(o1, o2))