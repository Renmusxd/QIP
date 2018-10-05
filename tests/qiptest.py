import unittest
from qip.qip import *
from qip.operators import *
from qip.pipeline import run


class TestQubitMethods(unittest.TestCase):

    def test_simple_pipeline(self):
        q = Qubit(n=1)
        n = Not(q)
        o, _ = n.run(feed={q: [1.0, 0.0]})
        self.assertTrue(numpy.allclose(o, [0., 1.]))

    def test_simple_pipeline_inv(self):
        q = Qubit(n=1)
        n = Not(q)
        o, _ = n.run(feed={q: [0.0, 1.0]})
        self.assertTrue(numpy.allclose(o, [1., 0.]))

    def test_bell(self):
        q = Qubit(n=1)
        h = H(q)
        o, _ = h.run(feed={q: [1.0, 0.0]})
        self.assertEqual(o[0], o[1])

    def test_h_first(self):
        q1 = Qubit(n=1)
        q2 = Qubit(n=1)
        h1 = H(q1)
        h2 = H(q2)
        q3, q4 = Qubit(h1, h2).split()
        h3 = H(q3)
        o1, _ = h3.run(feed={q1: [1.0, 0.0], q2: [1.0, 0.0]})

        # Should give same as H on just second qubit
        q1 = Qubit(n=1)
        q2 = Qubit(n=1)
        h3 = H(q2)
        q3, q4 = Qubit(q1, h3).split()
        o2, _ = q3.run(feed={q1: [1.0, 0.0], q2: [1.0, 0.0]})

        self.assertTrue(numpy.allclose(o1, o2))

    def test_bell_inv(self):
        q = Qubit(n=1)
        h = H(q)
        o, _ = h.run(feed={q: [0.0, 1.0]})
        self.assertEqual(o[0], -o[1])

    def test_multi_not(self):
        q1 = Qubit(n=1)
        q2 = Qubit(n=1)
        s = NotOp(q1,q2)
        o, _ = s.run(feed={q1: [1.0, 0.0], q2: [1.0, 0.0]})
        self.assertTrue(numpy.allclose(o, [0.0, 0.0, 0.0, 1.0]))

    def test_swap(self):
        q1 = Qubit(n=1)
        q2 = Qubit(n=1)
        s = SwapOp(q1,q2)
        o, _ = s.run(feed={q1: [1.0, 0.0], q2: [1.0, 0.0]})
        self.assertTrue(numpy.allclose(o, [1.0, 0.0, 0.0, 0.0]))

        o, _ = s.run(feed={q1: [1.0, 0.0], q2: [0.0, 1.0]})
        self.assertTrue(numpy.allclose(o, [0.0, 0.0, 1.0, 0.0]))

        o, _ = s.run(feed={q1: [0.0, 1.0], q2: [1.0, 0.0]})
        self.assertTrue(numpy.allclose(o, [0.0, 1.0, 0.0, 0.0]))

        o, _ = s.run(feed={q1: [0.0, 1.0], q2: [0.0, 1.0]})
        self.assertTrue(numpy.allclose(o, [0.0, 0.0, 0.0, 1.0]))

    def test_swap_twobit(self):
        q1 = Qubit(n=2)
        q2 = Qubit(n=2)
        s = SwapOp(q1,q2)
        o, _ = s.run(feed={q1: [0.0, 1.0, 0.0, 0.0], q2: [1.0, 0.0, 0.0, 0.0]})
        self.assertEqual(o[1], 1.0)
        self.assertEqual(sum(abs(o)), 1.0)

    def test_split(self):
        q1 = Qubit(n=1)
        q2 = Qubit(n=1)
        n1, n2 = Not(q1, q2)
        f1 = Not(n1)
        f2 = Not(n2)
        # f1 should be back to q1 whereas n2 should be negated.
        o, _ = f1.run(feed={q1: [1.0, 0.0], q2: [1.0, 0.0]})
        self.assertTrue(numpy.allclose(o, [0.0, 1.0, 0.0, 0.0]))

        # Vice versa
        o, _ = f2.run(feed={q1: [1.0, 0.0], q2: [1.0, 0.0]})
        self.assertTrue(numpy.allclose(o, [0.0, 0.0, 1.0, 0.0]))

        # Now both back to normal
        o, _ = run(f1, f2, feed={q1: [1.0, 0.0], q2: [1.0, 0.0]})
        self.assertTrue(numpy.allclose(o, [1.0, 0.0, 0.0, 0.0]))

    def test_allsplit(self):
        n = 5
        q = Qubit(n=n)
        qs = q.split(range(n+1))
        self.assertEqual(len(qs), n)
        newq = Qubit(*qs)

        state = numpy.cos(numpy.linspace(0,numpy.pi,2**n))
        o, _ = run(newq, feed={q: state})
        self.assertTrue(numpy.allclose(state, o))

    def test_cnot(self):
        q1 = Qubit(n=1)
        q2 = Qubit(n=1)
        c1, c2 = C(Not)(q1, q2)
        o, _ = run(c1, c2, feed={q1: [1.0, 0.0], q2: [1.0, 0.0]})
        self.assertTrue(numpy.allclose(o, [1.0, 0.0, 0.0, 0.0]))
        o,_ = run(c1, c2, feed={q1: [0.0, 1.0], q2: [1.0, 0.0]})
        self.assertTrue(numpy.allclose(o, [0.0, 0.0, 0.0, 1.0]))

    def test_default(self):
        # Default value setup
        q1 = Qubit(n=1, default=[0.0, 1.0])
        q2 = Qubit(n=1)
        o1, _ = run(q1, q2, feed={q2: [1.0, 0.0]})
        # Feed setup
        q3 = Qubit(n=1)
        q4 = Qubit(n=1)
        o2, _ = run(q3, q4, feed={q3: [0.0, 1.0], q4: [1.0, 0.0]})
        self.assertTrue(numpy.allclose(o1, o2))

    def test_default2(self):
        n = 2
        reg11 = Qubit(n=n, default=[1, 2, 3, 4])
        reg21 = Qubit(n=n, default=[1, 0, 0, 0])
        o1, _ = run(reg11, reg21)

        reg12 = Qubit(n=n, default=[1, 2, 3, 4])
        reg22 = Qubit(n=n)
        o2, _ = run(reg12, reg22, feed={reg22: [1, 0, 0, 0]})

        reg13 = Qubit(n=n)
        reg23 = Qubit(n=n)
        o3, _ = run(reg13, reg23, feed={reg13: [1, 2, 3, 4], reg23: [1, 0, 0, 0]})

        self.assertTrue(numpy.allclose(o1, o2))
        self.assertTrue(numpy.allclose(o1, o3))

    def test_cswap_compare(self):
        q1 = Qubit(n=1)
        q2 = Qubit(n=1)
        q3 = Qubit(n=1)

        h1 = H(q1)
        c1, c2, c3 = C(Swap)(h1, q2, q3)
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
        c1, c2, c3 = C(Swap)(h1, q2, q3)
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

        c1, c2, c3 = C(Swap)(h1, q2, q3)
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
        o1, _ = run(n1, n2, n3, feed={q1: [1.0, 0.0], q2: [1.0, 0.0],
                                      q3: [1.0, 0.0, 0.0, 0.0,
                                           0.0, 0.0, 0.0, 0.0]})
        q1 = Qubit(n=1)
        q2 = Qubit(n=1)
        q3 = Qubit(n=3)

        n2 = H(q2)
        n3 = Not(q3)
        n1 = Not(q1)
        o2, _ = run(n1, n2, n3, feed={q1: [1.0, 0.0], q2: [1.0, 0.0],
                                      q3: [1.0, 0.0, 0.0, 0.0,
                                           0.0, 0.0, 0.0, 0.0]})

        self.assertTrue(numpy.allclose(o1, o2))

    def test_cswap_measure(self):
        q1 = Qubit(n=1)
        q2 = Qubit(n=2)
        q3 = Qubit(n=2)

        h1 = H(q1)
        c1, c2, c3 = C(Swap)(h1, q2, q3)
        mh1 = H(c1)

        m1 = Measure(mh1)

        # CSwap compare should give P(|q1=0>) = 1/2 + 1/2<q2|q3> = 1/2
        o, classic = m1.run(feed={q1: [1.0, 0.0], q2: [0.0, 0.0, 0.0, 1.0], q3: [1.0, 0.0, 0.0, 0.0]})
        measured, measured_prob = classic[m1]

        self.assertAlmostEqual(measured_prob, 0.5)

    def test_cswap_reorder_measure(self):
        # Double check that indices aren't being messed up.
        q2 = Qubit(n=2)
        q1 = Qubit(n=1)
        q3 = Qubit(n=2)

        h1 = H(q1)
        c1, c2, c3 = C(Swap)(h1, q2, q3)
        mh1 = H(c1)

        m1 = Measure(mh1)

        # CSwap compare should give P(|q1=0>) = 1/2 + 1/2<q2|q3> = 1/2
        o, classic = m1.run(feed={q1: [1.0, 0.0], q2: [0.0, 0.0, 0.0, 1.0], q3: [1.0, 0.0, 0.0, 0.0]})
        measured, measured_prob = classic[m1]

        self.assertAlmostEqual(measured_prob, 0.5)

    def test_measure_stochastic(self):
        q1 = Qubit(n=2)
        q2 = Qubit(n=2)
        m1 = StochasticMeasure(q2)
        _, c = run(m1, q1, feed={q1: [0.0, 0.0, 1.0, 0.0],
                                 q2: [0.5, 0.5, 0.5, 0.5]})
        self.assertTrue(numpy.allclose(c[m1], [0.25, 0.25, 0.25, 0.25]))

    def test_measure_stochastic_top(self):
        q1 = Qubit(n=2)
        q2 = Qubit(n=2)
        m1 = StochasticMeasure(q2, top_k=3)
        _, c = run(m1, q1, feed={q1: [0.0, 0.0, 1.0, 0.0],
                                 q2: [0.6, 0.8, 0.0, 0.0]})

        self.assertEqual(c[m1][0][0], 1)
        self.assertEqual(c[m1][0][1], 0)
        # Doesn't matter what third index is since it's a tie.

        self.assertTrue(numpy.allclose(c[m1][1], [0.64, 0.36, 0.0]))

    def test_fop(self):
        n = 2
        reg1 = Qubit(n=n, default=[1, 2, 3, 4])
        reg2 = Qubit(n=n, default=[1, 0, 0, 0])
        freg1, freg2 = F(lambda x: 1, reg1, reg2)
        o1, _ = run(freg1, freg2)

        mock1 = Qubit(n=n, default=[1, 2, 3, 4])
        mock2 = Qubit(n=n, default=[0.0, 1.0, 0, 0])
        o2, _ = run(mock1, mock2)

        self.assertTrue(numpy.allclose(o1, o2))

    def test_fop_withq(self):
        # reg2 = 1 and f(x) = 1 therefore f(x)^reg2 == 0
        n = 2
        reg1 = Qubit(n=n, default=[1, 2, 3, 4])
        reg2 = Qubit(n=n, default=[0, 1, 0, 0])
        freg1, freg2 = F(lambda x: 1, reg1, reg2)
        o1, _ = run(freg1, freg2)

        mock1 = Qubit(n=n, default=[1, 2, 3, 4])
        mock2 = Qubit(n=n, default=[1, 0, 0, 0])
        o2, _ = run(mock1, mock2)

        self.assertTrue(numpy.allclose(o1, o2))

    def test_fop_regsize(self):
        n = 2
        reg1 = Qubit(n=2*n, default=numpy.ones(2**(2*n)) * pow(2**(2*n), -0.5))
        reg2 = Qubit(n=n)
        freg1, freg2 = F(lambda x: pow(3, x, 2**n), reg1, reg2)
        o1, _ = run(freg1, freg2)

    def test_basic_rop(self):
        q1 = Qubit(n=1)
        q2 = Qubit(n=1)
        r2 = Rm(2, q2)
        o, _ = run(q1, r2)
        self.assertTrue(numpy.allclose(numpy.abs(o), [1, 0, 0, 0]))

    def test_rop(self):
        q1 = Qubit(n=1)
        q2 = Qubit(n=1, default=1)
        r2 = Rm(2, q2)
        o, _ = run(q1, r2)
        self.assertTrue(numpy.allclose(numpy.abs(o), [0, 1.0, 0, 0]))

    def test_toffoli(self):
        q1 = Qubit(n=1)
        q2 = Qubit(n=1)
        q3 = Qubit(n=1)

        # CCNOT
        c1, c2, c3 = C(C(Not))(q1, q2, q3)

        o, c = run(c1, c2, c3, feed={q1: [pow(2, -0.5), pow(2, -0.5)],
                                     q2: [pow(2, -0.5), pow(2, -0.5)]})
        self.assertTrue(numpy.allclose(numpy.abs(o), [0.5, 0, 0.5, 0, 0.5, 0, 0, 0.5]))

    def test_rop_tuplefeed(self):
        q1 = Qubit(n=1)
        q2 = Qubit(n=1)
        r2 = Rm(2, q2)
        o1, _ = run(q1, r2, feed={q1: [1.0, 0.0], q2: [0.0, 1.0]})
        o2, _ = run(q1, r2, feed={(q1, q2): [0.0, 1.0, 0.0, 0.0]})
        o3, _ = run(q1, r2, feed={(q2, q1): [0.0, 0.0, 1.0, 0.0]})

        self.assertTrue(numpy.allclose(numpy.abs(o1), [0, 1.0, 0, 0]))
        self.assertTrue(numpy.allclose(numpy.abs(o2), [0, 1.0, 0, 0]))
        self.assertTrue(numpy.allclose(numpy.abs(o3), [0, 1.0, 0, 0]))


if __name__ == '__main__':
    unittest.main()
