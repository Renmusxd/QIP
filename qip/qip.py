from .pipeline import run
import numpy
from qip.ext.kronprod import measure
from qip.util import flatten


class PipelineObject(object):
    def __init__(self, quantum):
        self.quantum = quantum
        self.inputs = []
        self.sink = []

    def run(self, state=None, feed=None, **kwargs):
        return run(self, state=state, feed=feed, **kwargs)

    def feed(self, inputvals, qbitindex, n, arena):
        """
        Operate on the state of the system.
        :param inputvals: 2**n complex values
        :param qbitindex: mapping of qbit to global index
        :param n: number of qubits
        :param arena: memory arena to use for computations, must be of size 2**n
        :return: (2**n complex values of applying Q to input, memory arena of size 2**n, (num classic bits, int bits))
        """
        # Check to make sure enough are given
        if len(inputvals) != 2 ** n:
            raise Exception("Incorrect #inputs given: {} versus expected {}".format(len(inputvals), 2 ** n))
        # Return identity
        return inputvals, arena, (0, 0)

    def select_index(self, indices):
        """
        May be overridden to modify index selection (see SplitQubit).
        :param indices: list of indices from all inputs nodes
        :return:
        """
        return indices

    def remap_index(self, index_map, n):
        """
        May be override to rearrange qubits if needed
        :param index_map: map of Qubit -> Index
        :param n: n qubits in system
        :return: new index_map
        """
        return index_map


class Qubit(PipelineObject):
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
        super().__init__(True)
        if n is None:
            n = sum(q.n for q in inputs)
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
        for i in range(len(indexes) - 1):
            startn = indexes[i]
            endn = indexes[i + 1]
            sq = SplitQubit(list(range(startn, endn)), self, qid=qid)
            qid = sq.qid
            qs.append(sq)

        return tuple(qs)

    def extract_index(self, indices):
        """
        Select a set of indices out of a larger group
        :param indices: a list of indices (or single int) to select from the total. All indices must be 0 <= i < n
        :return: q1, q2 where q1 represents the selected indices and q2 the remaining.
        """
        if indices == int:
            indices = [indices]

        sq = SplitQubit([i for i in indices if 0 <= i < self.n], self)
        qs = SplitQubit([i for i in range(self.n) if i not in indices and 0 <= i < self.n], self, qid=sq.qid)
        return sq, qs

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
    def __init__(self, indices, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)
        self.indices = indices

    def select_index(self, indices):
        return [indices[i] for i in self.indices]


class Measure(PipelineObject):
    """
    Measures some quantum input.

    """

    def __init__(self, *inputs, measure_by=None, nosink=False):
        super().__init__(False)
        self.inputs = inputs
        self.n = sum(q.n for q in self.inputs)
        self.measure = measure_by

        if not nosink:
            for item in inputs:
                item.set_sink(self)

    def feed(self, inputvals, qbitindex, n, arena):
        """
        Operate on the state of the system.
        :param inputvals: 2**n complex values
        :param qbitindex: mapping of qbit to global index
        :param n: number of qubits
        :param arena: memory arena to use for computations, must be of size 2**n
        :return: (2**(n-m) complex values of applying Q to input, memory arena of size 2**(n-m), (m, bits)) where
            m is the number of qubits being measured.
        """
        # Get indices and make measurement
        indices = numpy.array(flatten(qbitindex[q] for q in self.inputs), dtype=numpy.int32)
        bits = measure(indices, n, inputvals, arena)

        # Cut and kill old memory after measurement so that footprint never grows above original.
        tmp_size = inputvals.shape[0]
        tmp_dtype = inputvals.dtype
        del inputvals

        new_inputvals = numpy.ndarray(shape=(tmp_size >> self.n), dtype=tmp_dtype)

        # Copy out relevant area from old arena
        new_arena = arena[:arena.shape[0] >> self.n]
        del arena

        return new_arena, new_inputvals, (self.n, bits)

    def remap_index(self, index_map, n):
        """
        May be override to rearrange qubits if needed
        :param index_map: map of Qubit -> Index
        :return: new index_map
        """
        removed_indices = list(sorted(index_map[q] for q in self.inputs))

        index_to_index_map = {}
        removed_so_far = 0
        for i in range(n):
            index_to_index_map[i] = i - removed_so_far
            if i == removed_indices[removed_so_far]:
                removed_so_far += 1

        return {q: [index_to_index_map[i] for i in index_map[q]] for q in index_map}

    def __repr__(self):
        return "M({})".format(",".join(i.__repr__() for i in self.inputs))