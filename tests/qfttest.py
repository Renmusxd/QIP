import unittest
from qip.qip import Qubit, Measure, StochasticMeasure
from qip.operators import *
from qip.qfft import *
from qip.pipeline import run

from matplotlib import pyplot


class TestQfftMethods(unittest.TestCase):

    def test_simple_pipeline(self):
        q = Qubit(n=2)
        qft = QFFT(q)
        o, _ = qft.run(feed={q: [0.5, 0.5, 0.5, 0.5]})
        print(o)
        self.assertTrue(numpy.allclose(o, [1., 0., 0., 0.]))

    def test_cosine_pipeline(self):
        freq = 5
        n = 6
        q = Qubit(n=n)
        qft = QFFT(q)

        state = numpy.cos(numpy.linspace(0,2*freq*numpy.pi,2**n))
        state = state / numpy.linalg.norm(state)
        o, _ = qft.run(feed={q: state})

        out = numpy.power(numpy.abs(o),2)
        self.assertTrue(numpy.argmax(out[:int(len(o)/2)]) == freq)


if __name__ == '__main__':
    unittest.main()
