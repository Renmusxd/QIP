from qip.pipeline import PipelineObject
from qip.util import flatten
from qip.backend import StateType, InitialState
import numpy
from typing import Sequence, Mapping, Tuple, Any, Callable, Union, Optional, Iterable, Type, cast


class Qubit(PipelineObject):
    """QIDs are used to ensure the no-cloning theorem holds.

    Applying a qubit to another input is akin to performing the identity transform.
    """
    QID = 0

    def __init__(self, *inputs: 'Qubit', n: int = None, qid: int =None,
                 nosink: bool = False, default: InitialState = None, quantum: bool = True, **kwargs):
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

    def split(self, indices: Optional[Iterable[int]] = None) -> Tuple['Qubit', ...]:
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
            # No need to wrap a SplitQubit around a single output.
            if len(indices) == 1:
                return self,
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
            qs.append(sq)
            qid = sq.qid

        return tuple(qs)

    def extract_index(self, indices: Union[int, Iterable[int]]):
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

    def set_sink(self, sink: 'PipelineObject'):
        if type(sink) == Qubit:
            sink = cast(Qubit, sink)
            if len(self.sink) == 0 or self.sink[0].qid == sink.qid:
                self.sink += [sink]
            else:
                raise Exception("Qubits may only sink to one output (no cloning)")
        else:
            self.sink += [sink]

    def __repr__(self):
        return "Q({})".format(self.qid)


class OpConstructor:
    def __init__(self, op: Union[Type[Qubit], Callable[[Sequence[Any]], Qubit]]):
        self.op = op

    def __call__(self, *args, **kwargs) -> Qubit:
        return self.op(*args, **kwargs)

    def wrap_op_hook(self, opconstructor: Callable[['OpConstructor'], 'OpConstructor'],
                     consumed_inputs: Optional[Sequence[int]]=None) -> Optional['OpConstructor']:
        return None

    def __repr__(self):
        return "Wrap({})".format(repr(self.op))


Q = OpConstructor(Qubit)


class SplitQubit(Qubit):
    def __init__(self, indices: Sequence[int], *inputs: Qubit, **kwargs):
        super().__init__(*inputs, n=len(indices), **kwargs)
        self.indices = indices

    def select_index(self, indices: Sequence[int]) -> Sequence[int]:
        return [indices[i] for i in self.indices]

    def __repr__(self) -> str:
        return "SplitQubit({})".format(",".join(map(repr, self.inputs)))


class Measure(Qubit):
    """Measures some quantum input."""

    def __init__(self, *inputs: Qubit, measure_by: int = None, nosink: bool = False):
        super().__init__(*inputs, nosink=nosink)
        self.inputs = inputs
        self.n = sum(q.n for q in self.inputs)
        self.measure = measure_by

        if not nosink:
            for item in inputs:
                item.set_sink(self)

    def feed_indices(self, state: StateType, index_groups: Sequence[Sequence[int]],
                     n: int) -> Tuple[int, Any]:
        # Get indices and make measurement
        indices = numpy.array(flatten(index_groups), dtype=numpy.int32)
        bits, prob = state.measure(indices)

        # We measure but don't remove any qubits.
        return 0, (bits, prob)

    def __repr__(self):
        return "M({})".format(",".join(i.__repr__() for i in self.inputs))


class StochasticMeasure(Qubit):
    """
    Measures some quantum input. Outputs the probability distribution
    as though the measurement was carried out repeatedly.
    If top_k is set then outputs a tuple with indices and probabilities for top k measurements.
    Does not change state.
    """

    def __init__(self, *inputs: Qubit, nosink: bool = False, top_k: int = 0):
        super().__init__(*inputs, quantum=False, nosink=nosink)
        self.top_k = top_k

        if not nosink:
            for item in inputs:
                item.set_sink(self)

    def feed_indices(self, state: StateType, index_groups: Sequence[Sequence[int]],
                     n: int) -> Tuple[int, object]:
        # Get indices and make measurement
        indices = numpy.array(flatten(index_groups), dtype=numpy.int32)
        probs = state.measure_probabilities(indices, top_k=self.top_k)[:]
        return 0, probs

    def __repr__(self) -> str:
        return "SM({})".format(",".join(i.__repr__() for i in self.inputs))