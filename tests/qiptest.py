import unittest
from qip.qip import Qubit
from qip.operators import *
from qip.pipeline import run


class TestQubitMethods(unittest.TestCase):

    def test_simple_pipeline(self):
        q = Qubit(n=1)
        n = Not(q)
        o = n.run(feed={q: [1.0, 0.0]})
        self.assertTrue(numpy.array_equal(o, [0., 1.]))

    def test_simple_pipeline_inv(self):
        q = Qubit(n=1)
        n = Not(q)
        o = n.run(feed={q: [0.0, 1.0]})
        self.assertTrue(numpy.array_equal(o, [1., 0.]))

    def test_bell(self):
        q = Qubit(n=1)
        h = H(q)
        o = h.run(feed={q: [1.0, 0.0]})
        self.assertEqual(o[0], o[1])

    def test_bell_inv(self):
        q = Qubit(n=1)
        h = H(q)
        o = h.run(feed={q: [0.0, 1.0]})
        self.assertEqual(o[0], -o[1])

    def test_multi_not(self):
        q1 = Qubit(n=1)
        q2 = Qubit(n=1)
        s = Not(q1,q2)
        o = s.run(feed={q1: [1.0, 0.0], q2: [1.0, 0.0]})
        self.assertTrue(numpy.array_equal(o, [0.0, 0.0, 0.0, 1.0]))

    def test_swap(self):
        q1 = Qubit(n=1)
        q2 = Qubit(n=1)
        s = Swap(q1,q2)
        o = s.run(feed={q1: [1.0, 0.0], q2: [1.0, 0.0]})
        self.assertTrue(numpy.array_equal(o, [1.0, 0.0, 0.0, 0.0]))

        o = s.run(feed={q1: [1.0, 0.0], q2: [0.0, 1.0]})
        self.assertTrue(numpy.array_equal(o, [0.0, 0.0, 1.0, 0.0]))

        o = s.run(feed={q1: [0.0, 1.0], q2: [1.0, 0.0]})
        self.assertTrue(numpy.array_equal(o, [0.0, 1.0, 0.0, 0.0]))

        o = s.run(feed={q1: [0.0, 1.0], q2: [0.0, 1.0]})
        self.assertTrue(numpy.array_equal(o, [0.0, 0.0, 0.0, 1.0]))

    def test_swap_twobit(self):
        q1 = Qubit(n=2)
        q2 = Qubit(n=2)
        s = Swap(q1,q2)
        o = s.run(feed={q1: [0.0, 1.0, 0.0, 0.0], q2: [1.0, 0.0, 0.0, 0.0]})
        self.assertEqual(o[1],1.0)
        self.assertEqual(sum(o),1.0)


if __name__ == '__main__':
    unittest.main()
