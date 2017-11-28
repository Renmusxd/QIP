import unittest
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
        self.assertEqual(o[1], 1.0)
        self.assertEqual(sum(abs(o)), 1.0)

    def test_split(self):
        q1 = Qubit(n=1)
        q2 = Qubit(n=1)
        n = Not(q1, q2)
        n1, n2 = n.split()
        f1 = Not(n1)
        f2 = Not(n2)
        # f1 should be back to q1 whereas n2 should be negated.
        o = f1.run(feed={q1: [1.0, 0.0], q2: [1.0, 0.0]})
        self.assertTrue(numpy.array_equal(o, [0.0, 1.0, 0.0, 0.0]))

        # Vice versa
        o = f2.run(feed={q1: [1.0, 0.0], q2: [1.0, 0.0]})
        self.assertTrue(numpy.array_equal(o, [0.0, 0.0, 1.0, 0.0]))

        # Now both back to normal
        o = run(f1,f2, feed={q1: [1.0, 0.0], q2: [1.0, 0.0]})
        self.assertTrue(numpy.array_equal(o, [1.0, 0.0, 0.0, 0.0]))

    def test_cnot(self):
        q1 = Qubit(n=1)
        q2 = Qubit(n=1)
        c = C(Not)(q1, q2)
        o = c.run(feed={q1: [1.0, 0.0], q2: [1.0, 0.0]})
        self.assertTrue(numpy.array_equal(o, [1.0, 0.0, 0.0, 0.0]))
        o = c.run(feed={q1: [0.0, 1.0], q2: [1.0, 0.0]})
        self.assertTrue(numpy.array_equal(o, [0.0, 0.0, 0.0, 1.0]))

    def test_cswap_compare(self):
        q1 = Qubit(n=1)
        q2 = Qubit(n=1)
        q3 = Qubit(n=1)

        h1 = H(q1)
        c1, c2, c3 = C(Swap)(h1, q2, q3).split()
        m1 = H(c1)

        # CSwap compare should give P(|q1=0>) = 1/2 + 1/2<q2|q3>
        o = m1.run(feed={q1: [1.0, 0.0], q2: [0.0, 1.0], q3: [1.0, 0.0]})

        # Since all real values, and control is index 1:
        p = numpy.dot(o[:4], o[:4])
        self.assertLessEqual(abs(p - 0.5), 1e-15)

        # CSwap compare should give P(|q1=0>) = 1/2 + 1/2<q2|q3>
        o = m1.run(feed={q1: [1.0, 0.0], q2: [1.0, 0.0], q3: [1.0, 0.0]})

        # Since all real values, and control is index 1:
        p = numpy.dot(o[:4], o[:4])
        self.assertLessEqual(abs(p - 1.0), 1e-15)

    def test_cswap_twobitcompare(self):
        q1 = Qubit(n=1)
        q2 = Qubit(n=2)
        q3 = Qubit(n=2)

        h1 = H(q1)
        c1, c2, c3 = C(Swap)(h1, q2, q3).split()
        m1 = H(c1)

        # CSwap compare should give P(|q1=0>) = 1/2 + 1/2<q2|q3>
        o = m1.run(feed={q1: [1.0, 0.0], q2: [0.0, 0.0, 0.0, 1.0], q3: [1.0, 0.0, 0.0, 0.0]})

        # Since all real values, and control is index 1:
        p = numpy.dot(o[:16], o[:16])
        self.assertLessEqual(abs(p - 0.5), 1e-15)

        # CSwap compare should give P(|q1=0>) = 1/2 + 1/2<q2|q3>
        o = m1.run(feed={q1: [1.0, 0.0], q2: [1.0, 0.0, 0.0, 0.0], q3: [1.0, 0.0, 0.0, 0.0]})

        # Since all real values, and control is index 1:
        p = numpy.dot(o[:16], o[:16])
        self.assertLessEqual(abs(p - 1.0), 1e-15)


if __name__ == '__main__':
    unittest.main()
