import unittest
import numpy
from qip.util import *
from qip.ext.func_apply import func_apply


class TestQubitMethods(unittest.TestCase):

    def test_simple_kron(self):
        i = numpy.array([[1, 0], [0, 1]])
        h = numpy.array([[1, 1], [1, -1]])
        hh = numpy.kron(h, h)

        ref = numpy.kron(h, numpy.kron(i, h))

        for i in range(ref.shape[0]):
            inputtest = numpy.zeros(shape=ref.shape[0], dtype=numpy.complex128)
            test1 = numpy.array(inputtest, dtype=numpy.complex128)
            test2 = numpy.array(inputtest, dtype=numpy.complex128)
            inputtest[i] = 1
            expect = ref[:,i]
            kronselect_dot({0: h, 2: h}, inputtest, 3, test1)
            kronselect_dot({(0, 2): hh}, inputtest, 3, test2)
            self.assertTrue(numpy.array_equal(expect, test1))
            self.assertTrue(numpy.array_equal(expect, test2))

    def test_nonsym_kron(self):
        i = numpy.array([[1, 0], [0, 1]])
        h = numpy.array([[1, 2], [3, 4]])
        hh = numpy.kron(h, h)

        ref = numpy.kron(h, numpy.kron(i, h))

        for i in range(ref.shape[0]):
            inputtest = numpy.zeros(shape=ref.shape[0], dtype=numpy.complex128)
            test1 = numpy.array(inputtest, dtype=numpy.complex128)
            test2 = numpy.array(inputtest, dtype=numpy.complex128)
            inputtest[i] = 1
            expect = ref[:,i]
            kronselect_dot({0: h, 2: h}, inputtest, 3, test1)
            kronselect_dot({(0, 2): hh}, inputtest, 3, test2)
            self.assertTrue(numpy.array_equal(expect, test1))
            self.assertTrue(numpy.array_equal(expect, test2))

    def test_nonsym_kron_rev(self):
        i = numpy.array([[1, 0], [0, 1]])
        a = numpy.array([[1, 2], [3, 4]])
        b = numpy.array([[5, 6], [7, 8]])
        ab = numpy.kron(a, b)
        ba = numpy.kron(b, a)

        ref = numpy.kron(a, numpy.kron(i, b))

        for i in range(ref.shape[0]):
            inputtest = numpy.zeros(shape=ref.shape[0], dtype=numpy.complex128)
            test1 = numpy.array(inputtest, dtype=numpy.complex128)
            test2 = numpy.array(inputtest, dtype=numpy.complex128)
            test3 = numpy.array(inputtest, dtype=numpy.complex128)
            inputtest[i] = 1
            expect = ref[:,i]
            kronselect_dot({0: a, 2: b}, inputtest, 3, test1)
            kronselect_dot({(0, 2): ab}, inputtest, 3, test2)
            kronselect_dot({(2, 0): ba}, inputtest, 3, test3)
            self.assertTrue(numpy.array_equal(expect, test1))
            self.assertTrue(numpy.array_equal(expect, test2))
            self.assertTrue(numpy.array_equal(expect, test3))

    def test_func_apply(self):
        xs = numpy.array([0,1], dtype=numpy.int32)
        ys = numpy.array([2,3], dtype=numpy.int32)
        n = len(xs) + len(ys)
        state = numpy.zeros(shape=(2**n,), dtype=numpy.complex128)
        output = numpy.array(state)

        # |0>|0>
        state[0] = 1
        # |1>|0>
        state[8] = 2
        # |2>|0>
        state[4] = 3
        # |3>|0>
        state[12] = 4

        func = lambda x: (x + 1) % 4
        func_apply(xs, ys, func, state, n, output)

        # |0>|1>
        self.assertEqual(output[1],1)
        # |1>|2>
        self.assertEqual(output[11], 2)
        # |2>|3>
        self.assertEqual(output[6], 3)
        # |3>|0>
        self.assertEqual(output[12], 4)


if __name__ == '__main__':
    unittest.main()
