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

    def test_bell(self):
        q = Qubit(n=1)
        h = H(q)
        o = h.run(feed={q: [1.0, 0.0]})
        self.assertEqual(o[0],o[1])

def test_vecequal(self,v1,v2):
    self.assertEqual(len(v1),len(v2),"Lengths must be the same: {}/{}".format(v1,v2))
    for i1, i2 in zip(v1,v2):
        self.assertEqual(i1,i2,"Contents must be equal: {}/{}".format(v1,v2))


if __name__ == '__main__':
    unittest.main()
