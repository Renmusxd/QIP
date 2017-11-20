import unittest
import numpy
from qip.util import *


class TestQubitMethods(unittest.TestCase):

    def test_simple_kron(self):
        i = numpy.array([[1, 0], [0, 1]])
        h = numpy.array([[1, 1], [1, -1]])
        hh = numpy.kron(h, h)

        ref = numpy.kron(h, numpy.kron(i, h))

        for i in range(ref.shape[0]):
            inputtest = numpy.zeros(shape=ref.shape[0])
            inputtest[i] = 1
            expect = ref[:,i]
            test1 = kronselect_dot({0: h, 2: h}, inputtest, 3)
            test2 = kronselect_dot({(0, 2): hh}, inputtest, 3)
            self.assertTrue(numpy.array_equal(expect, test1))
            self.assertTrue(numpy.array_equal(expect, test2))

    def test_nonsym_kron(self):
        i = numpy.array([[1, 0], [0, 1]])
        h = numpy.array([[1, 2], [3, 4]])
        hh = numpy.kron(h, h)

        ref = numpy.kron(h, numpy.kron(i, h))

        for i in range(ref.shape[0]):
            inputtest = numpy.zeros(shape=ref.shape[0])
            inputtest[i] = 1
            expect = ref[:,i]
            test1 = kronselect_dot({0: h, 2: h}, inputtest, 3)
            test2 = kronselect_dot({(0, 2): hh}, inputtest, 3)
            self.assertTrue(numpy.array_equal(expect, test1))
            self.assertTrue(numpy.array_equal(expect, test2))


if __name__ == '__main__':
    unittest.main()
