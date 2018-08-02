from qip.pipeline import run
from qip.util import flatten
from qip.backend import StateType, Backend
import numpy
from typing import List, Dict, Tuple, Type, Callable, Union, Optional

InitialState = Union[List[int], Tuple[int], int]

class OpConstructor:
    def __init__(self, op):
        self.op = op

    def __call__(self, *args, **kwargs) -> 'Qubit':
        return self.op(*args, **kwargs)

    def wrap_op_hook(self, opconstructor, consumed_inputs=None) -> Optional['Qubit']:
        return None


class PipelineObject(object):
    def __init__(self, quantum: bool, default: InitialState = None):
        self.quantum = quantum
        self.inputs = []
        self.sink = []
        self.default = default

    def run(self, state: InitialState =None, feed: Dict['PipelineObject', InitialState] = None, **kwargs):
        return run(self, state=state, feed=feed, **kwargs)

    def feed(self, inputvals: StateType, qbitindex: Dict['PipelineObject', List[int]], n: int, arena: StateType,
             backend: Backend) -> Tuple[StateType, StateType, Tuple[int, int]]:
        """
        Operate on the state of the system.
        :param inputvals: input state
        :param qbitindex: mapping of qbit to global index
        :param n: number of qubits
        :param arena: memory arena to use for computations
        :param backend: backend to use for matrix operations
        :return: (state, state_arena, (num classic bits, int bits))
        """
        return self.feed_indices(inputvals, [qbitindex[q] for q in self.inputs], n, arena, backend)

    def feed_indices(self, inputvals: StateType, index_groups: List[List[int]], n: int, arena: StateType,
                     backend: Backend) -> Tuple[StateType, StateType, Tuple[int, int]]:
        """
        Operate on the state of the system.
        :param inputvals: state
        :param index_groups: array of arrays of indicies used by each input in order.
        :param n: number of qubits
        :param arena: arena for state
        :param backend: backend to use for matrix operations
        :return: (state, state_arena, (num classic bits, int bits))
        """
        # Return identity
        return inputvals, arena, (0, 0)

    def select_index(self, indices: List[int]) -> List[int]:
        """
        May be overridden to modify index selection (see SplitQubit).
        :param indices: list of indices from all inputs nodes
        :return:
        """
        return indices

    def remap_index(self, index_map: Dict['PipelineObject', int], n: int) -> Dict['PipelineObject', int]:
        """
        May be override to rearrange qubits if needed
        :param index_map: map of Qubit -> Index
        :param n: n qubits in system
        :return: new index_map
        """
        return index_map

    def set_sink(self, sink: 'Qubit'):
        if type(sink) != Qubit or (len(self.sink) == 0 or self.sink[0].qid == sink.qid):
            self.sink += [sink]
        else:
            raise Exception("Qubits may only sink to one output (no cloning)")

    def get_inputs(self) -> List['PipelineObject']:
        return self.inputs

    def apply(self, op: OpConstructor) -> 'Qubit':
        return op(self)

    def __hash__(self):
        return hash(repr(self))


class Qubit(PipelineObject):
    """QIDs are used to ensure the no-cloning theorem holds.

    Applying a qubit to another input is akin to performing the identity transform.
    """
    QID = 0

    def __init__(self, *inputs, n=None, qid=None, nosink=False, default=None, quantum=True, **kwargs):
        """
        Create a qubit object
        :param inputs: inputs to qubit, qubit acts as identity on each
        :param n: number of qubits this object represents, set by inputs if inputs given, if not then 1.
        :param qid: forces qid value. For internal use or cloning.
        """
        super().__init__(quantum, default=default)
        if n is None:
            if len(inputs) > 0:
                n = sum(q.n for q in inputs)
            else:
                n = 1
        if type(default)==list and len(default) != 2**n:
            raise ValueError("Default state length must be 2**n")
        if n <= 0:
            raise Exception("Number of qubits must be greater than 0")

        self.n = n
        if qid is None:
            qid = Qubit.QID
            Qubit.QID += 1
        self.qid = qid
        self.inputs = inputs
        if not nosink:
            for item in inputs:
                item.set_sink(self)

    def split(self, indices=None):
        """
        Splits output qubits based on inputs.
        :return: n-tuple where n is the number of inputs
        """

        if indices is None:
            indices = []
            index = 0
            for qubit in self.inputs:
                indices.append(qubit.n + index)
                index += qubit.n
        else:
            indices = list(sorted(indices))

        if 0 not in indices:
            indices = [0] + indices
        if self.n not in indices:
            indices = indices + [self.n]

        qid = None
        qs = []
        for i in range(len(indices) - 1):
            startn = indices[i]
            endn = indices[i + 1]
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

    def __repr__(self):
        return "Q({})".format(self.qid)


Q = OpConstructor(Qubit)


class SplitQubit(Qubit):
    def __init__(self, indices, *inputs, **kwargs):
        super().__init__(*inputs, n=len(indices), **kwargs)
        self.indices = indices

    def select_index(self, indices):
        return [indices[i] for i in self.indices]

    def __repr__(self):
        return "SplitQubit({})".format(",".join(map(repr,self.inputs)))


class Measure(Qubit):
    """Measures some quantum input."""

    def __init__(self, *inputs, measure_by=None, nosink=False):
        super().__init__(*inputs, nosink=nosink)
        self.inputs = inputs
        self.n = sum(q.n for q in self.inputs)
        self.measure = measure_by

        if not nosink:
            for item in inputs:
                item.set_sink(self)

    def feed_indices(self, inputvals, index_groups, n, arena, backend):
        # Get indices and make measurement
        indices = numpy.array(flatten(index_groups), dtype=numpy.int32)
        bits = backend.measure(indices, n, inputvals, arena)

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


class StochasticMeasure(Qubit):
    """
    Measures some quantum input. Outputs the probability distribution
    as though the measurement was carried out repeatedly.
    Does not change state.
    """

    def __init__(self, *inputs, nosink=False):
        super().__init__(*inputs, quantum=False, nosink=nosink)

        if not nosink:
            for item in inputs:
                item.set_sink(self)

    def feed_indices(self, inputvals, index_groups, n, arena, backend):
        # Get indices and make measurement
        indices = numpy.array(flatten(index_groups), dtype=numpy.int32)
        probs = backend.measure_probabilities(indices, n, inputvals).copy()

        return inputvals, arena, (0, probs)

    def __repr__(self):
        return "SM({})".format(",".join(i.__repr__() for i in self.inputs))