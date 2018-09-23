from qip.ext.kronprod import cdot_loop
from qip.ext.kronprod import measure
from qip.ext.kronprod import soft_measure
from qip.ext.kronprod import measure_probabilities
from qip.ext.kronprod import prob_magnitude
from qip.ext.func_apply import func_apply
from qip.util import kronselect_dot, gen_edit_indices, InitialState, IndexType, MatrixType
import numpy
from typing import Sequence, Mapping, Callable, Optional, Tuple


class StateType:
    def __init__(self, n: int, state: object):
        self.n = n
        self.state = state

    def get_state(self) -> object:
        return self.state

    def kronselect_dot(self, mats: Mapping[IndexType, MatrixType],
                       input_offset: int = 0, output_offset: int = 0) -> None:
        raise NotImplemented("kronselect_dot not implemented by base class")

    def func_apply(self, reg1_indices: Sequence[int], reg2_indices: Sequence[int], func: Callable[[int],int],
                   input_offset: int = 0, output_offset: int = 0) -> None:
        raise NotImplemented("func_apply not implemented by base class")

    def total_prob(self) -> float:
        raise NotImplemented("total_prob not implemented by base class")

    def measure(self, indices: Sequence[int], measured: Optional[int] = None,
                measured_prob: Optional[float] = None,
                input_offset: int = 0, output_offset: int = 0) -> Tuple[int, float]:
        raise NotImplemented("measure not implemented by base class")

    def soft_measure(self, indices: Sequence[int], measured: Optional[int] = None,
                     input_offset: int = 0) -> Tuple[int, float]:
        raise NotImplemented("measure not implemented by base class")

    def measure_probabilities(self, indices: Sequence[int]) -> Sequence[float]:
        raise NotImplemented("measure_probabilities not implemented by base class")

    def close(self):
        pass


class CythonBackend(StateType):
    def __init__(self, n: int, state: numpy.ndarray, arena: numpy.ndarray):
        super().__init__(n, state)
        self.arena = arena

    @staticmethod
    def make_state(n: int, index_groups: Sequence[Sequence[int]], feed_list: Sequence[InitialState],
                   state: InitialState = None, 
                   inputstartindex: Optional[int] = None, inputendindex: Optional[int] = None,
                   outputstartindex: Optional[int] = None, outputendindex: Optional[int] = None,
                   statetype: type = numpy.complex128) -> 'CythonBackend':
        if state is None:
            if inputstartindex is None:
                inputstartindex = 0
            if inputendindex is None:
                inputendindex = 2**n
            if outputstartindex is None:
                outputstartindex = 0
            if outputendindex is None:
                outputendindex = 2**n

            state = numpy.zeros(inputendindex - inputstartindex, dtype=statetype)

            if len(feed_list) == 0:
                state[0] = 1

            # Set all the entries in state to product of matrix entries
            for index, flips in gen_edit_indices(index_groups, n - 1):
                # Skip out of range
                if index < inputstartindex or index >= inputendindex:
                    continue
                # Make index value
                state[index - inputstartindex] = 1.0
                for qindex, flip in enumerate(flips):
                    state[index - inputstartindex] = state[index - inputstartindex] * feed_list[qindex][flip]
        elif type(state) == int:
            stateint = state - inputstartindex
            state = numpy.zeros(2 ** n, dtype=statetype)
            if stateint > 0:
                state[stateint] = 1.0
        elif len(state) != 2 ** n:
            raise ValueError("State size must be 2**n")
        elif type(state) == list:
            state = numpy.array(state)

        arena = numpy.ndarray(outputendindex - outputstartindex, dtype=statetype)
        return CythonBackend(n, state, arena)

    def get_state(self) -> numpy.ndarray:
        return self.state

    def swap(self):
        self.state, self.arena = self.arena, self.state

    def kronselect_dot(self, mats: Mapping[IndexType, MatrixType],
                       input_offset: int = 0, output_offset: int = 0) -> None:
        kronselect_dot(mats, self.state, self.n, self.arena, dot_impl=cdot_loop,
                       input_offset=input_offset, output_offset=output_offset)
        self.state, self.arena = self.arena, self.state

    def func_apply(self, reg1_indices: Sequence[int], reg2_indices: Sequence[int], func: Callable[[int], int],
                   input_offset: int = 0, output_offset: int = 0) -> None:
        func_apply(reg1_indices, reg2_indices, func, self.state, self.n, self.arena)
        self.state, self.arena = self.arena, self.state

    def total_prob(self) -> float:
        return prob_magnitude(self.state)

    def measure(self, indices: Sequence[int], measured: Optional[int] = None,
                measured_prob: Optional[float] = None,
                input_offset: int = 0, output_offset: int = 0) -> Tuple[int, float]:
        # Get an appropriately sized arena
        numpy.resize(self.arena, (self.arena.shape[0] >> len(indices),))
        bits, prob = measure(numpy.asarray(indices, dtype=numpy.int32),
                             self.n, self.state, self.arena,
                             measured=measured, measured_prob=measured_prob,
                             input_offset=input_offset, output_offset=output_offset)
        numpy.resize(self.state, self.arena.shape)
        self.state, self.arena = self.arena, self.state

        return bits, prob

    def soft_measure(self, indices: Sequence[int], measured: Optional[int] = None,
                     input_offset: int = 0) -> Tuple[int, float]:
        # TODO tuple not acceptable?
        return soft_measure(numpy.asarray(indices, dtype=numpy.int32), self.n, self.state,
                            measured=measured, input_offset=input_offset)

    def measure_probabilities(self, indices: Sequence[int]) -> Sequence[float]:
        return measure_probabilities(numpy.asarray(indices, dtype=numpy.int32), self.n, self.state)
