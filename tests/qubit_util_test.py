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

    def test_function_wrapper(self):

        @QubitFuncWrapper.wrap
        def circuit(*args):
            return H(*args)

        q1 = Qubit(n=1, default=[pow(2, -0.5), pow(2, -0.5)])
        q2 = Qubit(n=2)

        c1, c2 = C(H)(q1, q2)
        o1, _ = run(c1, c2)

        q3 = Qubit(n=1, default=[pow(2, -0.5), pow(2, -0.5)])
        q4 = Qubit(n=2)

        c3, c4 = C(circuit)(q3, q4)
        o2, _ = run(c3, c4)

        self.assertTrue(numpy.allclose(o1, o2))

    def test_function_nested_c(self):

        @QubitFuncWrapper.wrap
        def circuit(*args):
            return C(H)(*args)

        q1 = Qubit(n=1, default=[pow(2, -0.5), pow(2, -0.5)])
        q2 = Qubit(n=1, default=[pow(2, -0.5), pow(2, -0.5)])
        q3 = Qubit(n=1)

        c1, c2, c3 = C(C(H))(q1, q2, q3)
        o1, _ = run(c1, c2, c3)

        q4 = Qubit(n=1, default=[pow(2, -0.5), pow(2, -0.5)])
        q5 = Qubit(n=1, default=[pow(2, -0.5), pow(2, -0.5)])
        q6 = Qubit(n=1)

        c4, c5, c6 = C(circuit)(q4, q5, q6)
        o2, _ = run(c4, c5, c6)

        self.assertTrue(numpy.allclose(o1, o2))

    def test_function_nested_wrapper(self):

        @QubitFuncWrapper.wrap
        def circuit_b(*args):
            return H(*args)

        @QubitFuncWrapper.wrap
        def circuit_a(*args):
            return C(circuit_b)(*args)

        q1 = Qubit(n=1, default=[pow(2, -0.5), pow(2, -0.5)])
        q2 = Qubit(n=1, default=[pow(2, -0.5), pow(2, -0.5)])
        q3 = Qubit(n=1)

        c1, c2, c3 = C(C(H))(q1, q2, q3)
        o1, _ = run(c1, c2, c3)

        q4 = Qubit(n=1, default=[pow(2, -0.5), pow(2, -0.5)])
        q5 = Qubit(n=1, default=[pow(2, -0.5), pow(2, -0.5)])
        q6 = Qubit(n=1)

        c4, c5, c6 = C(circuit_a)(q4, q5, q6)
        o2, _ = run(c4, c5, c6)

        self.assertTrue(numpy.allclose(o1, o2))

    def test_function_double_nested_c(self):

        @QubitFuncWrapper.wrap
        def circuit(*args):
            return C(C(H))(*args)

        q1 = Qubit(n=1, default=[pow(2, -0.5), pow(2, -0.5)])
        q2 = Qubit(n=1, default=[pow(2, -0.5), pow(2, -0.5)])
        q3 = Qubit(n=1, default=[pow(2, -0.5), pow(2, -0.5)])
        q4 = Qubit(n=1)

        c1, c2, c3, c4 = C(C(C(H)))(q1, q2, q3, q4)
        o1, _ = run(c1, c2, c3, c4)

        q5 = Qubit(n=1, default=[pow(2, -0.5), pow(2, -0.5)])
        q6 = Qubit(n=1, default=[pow(2, -0.5), pow(2, -0.5)])
        q7 = Qubit(n=1, default=[pow(2, -0.5), pow(2, -0.5)])
        q8 = Qubit(n=1)

        c5, c6, c7, c8 = C(circuit)(q5, q6, q7, q8)
        o2, _ = run(c5, c6, c7, c8)

        self.assertTrue(numpy.allclose(o1, o2))