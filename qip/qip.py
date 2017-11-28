from .pipeline import run
import numpy


class Qubit(object):
    """QIDs are used to ensure the no-cloning theorem holds.

    Applying a qubit to another input is akin to performing the identity transform.
    """
    QID = 0

    def __init__(self, *inputs, n=None, qid=None, nosink=False):
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
        if not nosink:
            for item in inputs:
                item.set_sink(self)

    def run(self, state=None, feed=None, **kwargs):
        return run(self, state=state, feed=feed, **kwargs)

    def feed(self, inputvals, qbitindex, n):
        """
        Operate on the state of the system.
        :param inputvals: 2**n complex values
        :param qbitindex: mapping of qbit to global index
        :return: 2**n complex values
        """
        # Check to make sure enough are given
        if len(inputvals) != 2**n:
            raise Exception("Incorrect #inputs given: {} versus expected {}".format(len(inputvals), 2**self.n))
        # Return identity
        return inputvals

    def split(self):
        """
        Splits output qubits based in inputs.
        :return: n-tuple where n is the number of inputs
        """
        qs = []
        n = 0
        qid = None
        for qbit in self.inputs:
            qs.append(SplitQubit(n, n+qbit.n, self, qid=qid))
            qid = qs[-1].qid
            n += qbit.n
        return tuple(qs)

    def select_index(self, indices):
        """
        May be overridden to modify index selection (see SplitQubit).
        :param indices: list of indices from all inputs nodes
        :return:
        """
        return indices

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

    @property
    def qid(self):
        return self._qid

    def __repr__(self):
        return "Q({})".format(self._qid)


class SplitQubit(Qubit):
    def __init__(self, startn, endn, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)
        self.startn = startn
        self.endn = endn

    def select_index(self, indices):
        return indices[self.startn:self.endn]
