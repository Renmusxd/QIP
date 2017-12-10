import unittest
from qip.qip import Qubit, Measure
from qip.operators import *
from qip.pipeline import run


class TestQubitMethods(unittest.TestCase):

    def test_simple_pipeline(self):
        q = Qubit(n=1)
        n = Not(q)
        o, _ = n.run(feed={q: [1.0, 0.0]})
        self.assertTrue(numpy.array_equal(o, [0., 1.]))

    def test_simple_pipeline_inv(self):
        q = Qubit(n=1)
        n = Not(q)
        o, _ = n.run(feed={q: [0.0, 1.0]})
        self.assertTrue(numpy.array_equal(o, [1., 0.]))

    def test_bell(self):
        q = Qubit(n=1)
        h = H(q)
        o, _ = h.run(feed={q: [1.0, 0.0]})
        self.assertEqual(o[0], o[1])

    def test_bell_inv(self):
        q = Qubit(n=1)
        h = H(q)
        o, _ = h.run(feed={q: [0.0, 1.0]})
        self.assertEqual(o[0], -o[1])

    def test_multi_not(self):
        q1 = Qubit(n=1)
        q2 = Qubit(n=1)
        s = Not(q1,q2)
        o, _ = s.run(feed={q1: [1.0, 0.0], q2: [1.0, 0.0]})
        self.assertTrue(numpy.array_equal(o, [0.0, 0.0, 0.0, 1.0]))

    def test_swap(self):
        q1 = Qubit(n=1)
        q2 = Qubit(n=1)
        s = Swap(q1,q2)
        o, _ = s.run(feed={q1: [1.0, 0.0], q2: [1.0, 0.0]})
        self.assertTrue(numpy.array_equal(o, [1.0, 0.0, 0.0, 0.0]))

        o, _ = s.run(feed={q1: [1.0, 0.0], q2: [0.0, 1.0]})
        self.assertTrue(numpy.array_equal(o, [0.0, 0.0, 1.0, 0.0]))

        o, _ = s.run(feed={q1: [0.0, 1.0], q2: [1.0, 0.0]})
        self.assertTrue(numpy.array_equal(o, [0.0, 1.0, 0.0, 0.0]))

        o, _ = s.run(feed={q1: [0.0, 1.0], q2: [0.0, 1.0]})
        self.assertTrue(numpy.array_equal(o, [0.0, 0.0, 0.0, 1.0]))

    def test_swap_twobit(self):
        q1 = Qubit(n=2)
        q2 = Qubit(n=2)
        s = Swap(q1,q2)
        o, _ = s.run(feed={q1: [0.0, 1.0, 0.0, 0.0], q2: [1.0, 0.0, 0.0, 0.0]})
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
        o, _ = f1.run(feed={q1: [1.0, 0.0], q2: [1.0, 0.0]})
        self.assertTrue(numpy.array_equal(o, [0.0, 1.0, 0.0, 0.0]))

        # Vice versa
        o, _ = f2.run(feed={q1: [1.0, 0.0], q2: [1.0, 0.0]})
        self.assertTrue(numpy.array_equal(o, [0.0, 0.0, 1.0, 0.0]))

        # Now both back to normal
        o, _ = run(f1, f2, feed={q1: [1.0, 0.0], q2: [1.0, 0.0]})
        self.assertTrue(numpy.array_equal(o, [1.0, 0.0, 0.0, 0.0]))

    def test_cnot(self):
        q1 = Qubit(n=1)
        q2 = Qubit(n=1)
        c = C(Not)(q1, q2)
        o, _ = c.run(feed={q1: [1.0, 0.0], q2: [1.0, 0.0]})
        self.assertTrue(numpy.array_equal(o, [1.0, 0.0, 0.0, 0.0]))
        o,_ = c.run(feed={q1: [0.0, 1.0], q2: [1.0, 0.0]})
        self.assertTrue(numpy.array_equal(o, [0.0, 0.0, 0.0, 1.0]))

    def test_cswap_compare(self):
        q1 = Qubit(n=1)
        q2 = Qubit(n=1)
        q3 = Qubit(n=1)

        h1 = H(q1)
        c1, c2, c3 = C(Swap)(h1, q2, q3).split()
        m1 = H(c1)

        # CSwap compare should give P(|q1=0>) = 1/2 + 1/2<q2|q3>
        o, _ = m1.run(feed={q1: [1.0, 0.0], q2: [0.0, 1.0], q3: [1.0, 0.0]})

        # Since all real values, and control is index 1:
        p = numpy.dot(o[:len(o) >> 1], o[:len(o) >> 1])
        self.assertLessEqual(abs(p - 0.5), 1e-15)

        # CSwap compare should give P(|q1=0>) = 1/2 + 1/2<q2|q3>
        o, _ = m1.run(feed={q1: [1.0, 0.0], q2: [1.0, 0.0], q3: [1.0, 0.0]})

        # Since all real values, and control is index 1:
        p = numpy.dot(o[:len(o) >> 1], o[:len(o) >> 1])
        self.assertLessEqual(abs(p - 1.0), 1e-15)

    def test_cswap_twobitcompare(self):
        q1 = Qubit(n=1)
        q2 = Qubit(n=2)
        q3 = Qubit(n=2)

        h1 = H(q1)
        c1, c2, c3 = C(Swap)(h1, q2, q3).split()
        m1 = H(c1)

        # CSwap compare should give P(|q1=0>) = 1/2 + 1/2<q2|q3> = 1/2
        o,_ = m1.run(feed={q1: [1.0, 0.0], q2: [0.0, 0.0, 0.0, 1.0], q3: [1.0, 0.0, 0.0, 0.0]})

        # Since all real values, and control is index 1:
        p = numpy.dot(o[:len(o) >> 1], o[:len(o) >> 1])
        self.assertLessEqual(abs(p - 0.5), 1e-15)

        # CSwap compare should give P(|q1=0>) = 1/2 + 1/2<q2|q3> = 1
        o, _ = m1.run(feed={q1: [1.0, 0.0], q2: [1.0, 0.0, 0.0, 0.0], q3: [1.0, 0.0, 0.0, 0.0]})

        # Since all real values, and control is index 1:
        p = numpy.dot(o[:len(o) >> 1], o[:len(o) >> 1])
        self.assertLessEqual(abs(p - 1.0), 1e-15)

    def test_cswap_5bitcompare(self):
        q1 = Qubit(n=1)
        q2 = Qubit(n=5)
        q3 = Qubit(n=5)

        h1 = H(q1)
        c1, c2, c3 = C(Swap)(h1, q2, q3).split()
        m1 = H(c1)

        # CSwap compare should give P(|q1=0>) = 1/2 + 1/2<q2|q3> = 1/2
        state2 = numpy.zeros((32,))
        state3 = numpy.zeros((32,))

        state2[0] = 1.0
        state3[1] = 1.0

        o, _ = m1.run(feed={q1: [1.0, 0.0], q2: state2, q3: state3})

        # Since all real values, and control is index 1:
        p = numpy.dot(o[:len(o) >> 1], o[:len(o) >> 1])
        self.assertLessEqual(abs(p - 0.5), 1e-15)

        state2 = numpy.zeros((32,))
        state3 = numpy.zeros((32,))

        state2[0] = 1.0
        state3[0] = 1.0

        # CSwap compare should give P(|q1=0>) = 1/2 + 1/2<q2|q3> = 1
        o, _ = m1.run(feed={q1: [1.0, 0.0], q2: state2, q3: state3})

        # Since all real values, and control is index 1:
        p = numpy.dot(o[:len(o) >> 1], o[:len(o) >> 1])
        self.assertLessEqual(abs(p - 1.0), 1e-15)

    def test_many_bits(self):
        q1 = Qubit(n=19)
        q2 = Qubit(n=1)
        n2 = Not(q2)
        o, _ = run(q1, n2, feed={q2: [1.0, 0.0]})
        self.assertEqual(o[1], numpy.sum(numpy.abs(o)))

    def test_not_ordered(self):
        q1 = Qubit(n=1)
        q2 = Qubit(n=1)
        q3 = Qubit(n=3)

        n1 = Not(q1)
        n2 = H(q2)
        n3 = Not(q3)
        o1, _ = run(n1,n2,n3, feed={q1: [1.0, 0.0], q2: [1.0, 0.0],
                                 q3: [1.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0]})

        q1 = Qubit(n=1)
        q2 = Qubit(n=1)
        q3 = Qubit(n=3)

        n2 = H(q2)
        n3 = Not(q3)
        n1 = Not(q1)
        o2, _ = run(n1,n2,n3, feed={q1: [1.0, 0.0], q2: [1.0, 0.0],
                                 q3: [1.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0]})

        self.assertTrue(numpy.array_equal(o1, o2))

    def test_cswap_measure(self):
        q1 = Qubit(n=1)
        q2 = Qubit(n=2)
        q3 = Qubit(n=2)

        h1 = H(q1)
        c1, c2, c3 = C(Swap)(h1, q2, q3).split()
        mh1 = H(c1)

        m1 = Measure(mh1)

        # CSwap compare should give P(|q1=0>) = 1/2 + 1/2<q2|q3> = 1/2
        o, classic = m1.run(feed={q1: [1.0, 0.0], q2: [0.0, 0.0, 0.0, 1.0], q3: [1.0, 0.0, 0.0, 0.0]})
        measured, measured_prob = classic[m1]

        self.assertAlmostEqual(measured_prob, 0.5)



if __name__ == '__main__':
    unittest.main()
