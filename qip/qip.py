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
        self.qid = qid
        self.inputs = inputs
        if not nosink:
            for item in inputs:
                item.set_sink(self)

    def run(self, state=None, feed=None, **kwargs):
        return run(self, state=state, feed=feed, **kwargs)

    def feed(self, inputvals, qbitindex, n, arena):
        """
        Operate on the state of the system.
        :param inputvals: 2**n complex values
        :param qbitindex: mapping of qbit to global index
        :param n: number of qubits
        :param arena: memory arena to use for computations, must be of size 2**n
        :return: (2**n complex values of applying Q to input, memory arena of size 2**n)
        """
        # Check to make sure enough are given
        if len(inputvals) != 2**n:
            raise Exception("Incorrect #inputs given: {} versus expected {}".format(len(inputvals), 2**self.n))
        # Return identity
        return inputvals, arena

    def split(self, indexes=None):
        """
        Splits output qubits based in inputs.
        :return: n-tuple where n is the number of inputs
        """

        index = 0
        if indexes is None:
            indexes = []
            for qubit in self.inputs:
                indexes.append(qubit.n + index)
                index += qubit.n

        indexes = [0] + indexes
        qid = None
        qs = []
        for i in range(len(indexes)-1):
            startn = indexes[i]
            endn = indexes[i+1]
            sq = SplitQubit(startn, endn, self, qid=qid)
            qid = sq.qid
            qs.append(sq)

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

    def __repr__(self):
        return "Q({})".format(self.qid)


class SplitQubit(Qubit):
    def __init__(self, startn, endn, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)
        self.startn = startn
        self.endn = endn

    def select_index(self, indices):
        return indices[self.startn:self.endn]
