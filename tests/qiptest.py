import unittest
from qip.qip import Qubit
from qip.operators import *
from qip.pipeline import run


class TestQubitMethods(unittest.TestCase):

    def test_simple_pipeline(self):
        q = Qubit(n=1)
        n = Not(q)
        o = n.run(feed={q: [1.0, 0.0]})
        test_vecequal(self, o, [0., 1.])

    def test_split(self):
        q = Qubit(n=2)
        q1, q2 = q.partition(1)
        self.assertEqual(q1.n, 1)
        self.assertEqual(q2.n, 1)

        n1 = Not(q1)
        i2 = q2

        o1, o2 = run(n1, i2, feed={q: [1, 0, 1, 0]})
        test_vecequal(self, o1, [0., 1.])
        test_vecequal(self, o2, [1., 0.])

    def test_resuse_pipeline(self):
        q = Qubit(n=3)
        q1, q2 = q.partition(1)
        i1 = q1
        n2 = Not(q2)

        o11, o12 = run(i1, n2, feed={q: [1, 0, 1, 0, 1, 0]})
        o21, o22 = run(i1, n2, feed={q: [0, 1, 0, 1, 0, 1]})
        test_vecequal(self, o11, [1, 0])
        test_vecequal(self, o12, [0, 1, 0, 1])
        test_vecequal(self, o21, [0, 1])
        test_vecequal(self, o22, [1, 0, 1, 0])

    def test_merge_qubits(self):
        q1 = Qubit(n=1)
        q2 = Qubit(n=2)
        q = Qubit(q1, q2)
        n = Not(q)
        o = run(n, feed={q1: [1, 0], q2: [0, 1, 0, 1]})
        test_vecequal(self, o, [0, 1, 1, 0, 1, 0])


    def test_swap(self):
        q1 = Qubit(n=1)
        q2 = Qubit(n=1)
        s = Swap(q1,q2)
        p1, p2 = s.partition(1)

        o1, o2 = run(p1, p2, feed={q1: [1,0], q2: [0,1]})
        test_vecequal(self, o1, [0, 1])
        test_vecequal(self, o2, [1, 0])

def test_vecequal(self,v1,v2):
    self.assertEqual(len(v1),len(v2),"Lengths must be the same: {}/{}".format(v1,v2))
    for i1, i2 in zip(v1,v2):
        self.assertEqual(i1,i2,"Contents must be equal: {}/{}".format(v1,v2))


if __name__ == '__main__':
    unittest.main()
