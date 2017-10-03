from .pipeline import run
import numpy


class Qubit(object):
    """QIDs are used to ensure the no-cloning theorem holds.

    Applying a qubit to another input is akin to performing the identity transform.
    """
    QID = 0

    def __init__(self, *inputs, n=None, qid=None):
        """
        Create a qubit object
        :param inputs: inputs to qubit, qubit acts as identity on each
        :param n: number of qubits this object represents, set by inputs if given
        :param qid: forces qid value. For internal use or cloning.
        """
        super().__init__()
        if n is None:
            n = sum(q.n for q in inputs)
        self.sink = []
        self.n = n
        self.check()
        if qid is None:
            qid = Qubit.QID
            Qubit.QID += 1
        self._qid = qid
        self.inputs = inputs
        for item in inputs:
            item.set_sink(self)

    def run(self, **kwargs):
        return run(self, **kwargs)

    def feed(self, inputvals):
        """
        Feeds values through qubit operator.
        :param inputvals: 2^n complex numbers for each of |x1 x2 ... xn>
        :return: 2^n complex numbers
        """
        # Check to make sure enough are given
        if len(inputvals) != 2**self.n:
            raise Exception("Incorrect #inputs given: {} versus expected {}".format(len(inputvals), 2**self.n))
        # Return identity
        return inputvals[:]

    def set_sink(self, sink):
        if len(self.sink) == 0 or self.sink[0].qid == sink.qid:
            self.sink += [sink]
        else:
            raise Exception("Qubits may only sink to one output (no cloning)")

    def get_inputs(self):
        return self.inputs

    def apply(self, op):
        return op(self)

    def check(self):
        if self.n <= 0:
            raise Exception("Number of qubits must be greater than 0")

    def partition(self, n):
        q1 = SplitQubit(0, n, self)
        q2 = SplitQubit(n, self.n-n+1, self, qid=q1.qid)
        return q1, q2

    @property
    def qid(self):
        return self._qid


class SplitQubit(Qubit):
    def __init__(self, startn, endn, *inputs, qid=None):
        super(SplitQubit,self).__init__(*inputs, n=endn-startn, qid=qid)
        self.startn = startn
        self.endn = endn

    def feed(self, inputvals):
        # Check to make sure enough are given
        if len(inputvals) < 2*self.n:
            raise Exception("Incorrect #inputs given: {} versus expected {}".format(len(inputvals), 2*self.n))
        return inputvals[2*self.startn:2*self.endn]
