from qip.ext.kronprod import cdot_loop
from qip.ext.kronprod import reduce_measure
from qip.ext.kronprod import measure
from qip.ext.kronprod import soft_measure
from qip.ext.kronprod import measure_probabilities
from qip.ext.kronprod import measure_top_probabilities
from qip.ext.kronprod import prob_magnitude
from qip.ext.func_apply import func_apply
from qip.util import kronselect_dot, gen_edit_indices, InitialState, IndexType, MatrixType
import numpy
from typing import Sequence, Mapping, Callable, Optional, Tuple, Union


class StateType(object):
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
        raise NotImplemented("reduce_measure not implemented by base class")

    def reduce_measure(self, indices: Sequence[int], measured: Optional[int] = None,
                       measured_prob: Optional[float] = None,
                       input_offset: int = 0, output_offset: int = 0) -> Tuple[int, float]:
        raise NotImplemented("reduce_measure not implemented by base class")

    def soft_measure(self, indices: Sequence[int], measured: Optional[int] = None,
                     input_offset: int = 0) -> Tuple[int, float]:
        raise NotImplemented("measure not implemented by base class")

    def measure_probabilities(self, indices: Sequence[int],
                              top_k: int = 0) -> Union[Sequence[float], Tuple[Sequence[int], Sequence[float]]]:
        """If top_k is 0 then output array of all probabilities, else tuple of indices, probs for top_k probs."""
        raise NotImplemented("measure_probabilities not implemented by base class")

    def get_state_size(self) -> int:
        raise NotImplemented("get_state_size not implemented by base class")

    def get_relative_range(self, start: int, end: int) -> Sequence[complex]:
        raise NotImplemented("get_relative_range not implemented by base class")

    def overwrite_relative_range(self, start: int, end: int, data: Sequence[complex]):
        raise NotImplemented("overwrite_relative_range not implemented by base class")

    def addto_relative_range(self, start: int, end: int, data: Sequence[complex]):
        raise NotImplemented("addto_relative_range not implemented by base class")

    def close(self):
        pass


class CythonBackend(StateType):
    def __init__(self, n: int, state: numpy.ndarray, arena: numpy.ndarray):
        super().__init__(n, state)
        self.arena = arena

    @staticmethod
    def make_state(n: int, index_groups: Sequence[Sequence[int]], feed_list: Sequence[InitialState],
                   inputstartindex: Optional[int] = None, inputendindex: Optional[int] = None,
                   outputstartindex: Optional[int] = None, outputendindex: Optional[int] = None,
                   statetype: type = numpy.complex128) -> 'CythonBackend':

        if inputstartindex is None:
            inputstartindex = 0
        if inputendindex is None:
            inputendindex = 2 ** n
        if outputstartindex is None:
            outputstartindex = 0
        if outputendindex is None:
            outputendindex = 2 ** n

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
                input_offset: int = 0, output_offset: int = 0):
        bits, prob = measure(numpy.asarray(indices, dtype=numpy.int32),
                             self.n, self.state, self.arena,
                             measured=measured, measured_prob=measured_prob,
                             input_offset=input_offset, output_offset=output_offset)
        self.state, self.arena = self.arena, self.state
        return bits, prob

    def reduce_measure(self, indices: Sequence[int], measured: Optional[int] = None,
                       measured_prob: Optional[float] = None,
                       input_offset: int = 0, output_offset: int = 0) -> Tuple[int, float]:
        # TODO Get an appropriately sized arena
        # new_arena_size = 2**(self.n - len(indices))
        # if new_arena_size < self.arena.shape[0]:
        #     self.arena.resize((new_arena_size,))

        bits, prob = reduce_measure(numpy.asarray(indices, dtype=numpy.int32),
                                    self.n, self.state, self.arena,
                                    measured=measured, measured_prob=measured_prob,
                                    input_offset=input_offset, output_offset=output_offset)
        # self.state.resize(self.arena.shape)
        self.state, self.arena = self.arena, self.state

        return bits, prob

    def soft_measure(self, indices: Sequence[int], measured: Optional[int] = None,
                     input_offset: int = 0) -> Tuple[int, float]:
        return soft_measure(numpy.asarray(indices, dtype=numpy.int32), self.n, self.state,
                            measured=measured, input_offset=input_offset)

    def measure_probabilities(self, indices: Sequence[int],
                              top_k: int = 0) -> Union[Sequence[float], Tuple[Sequence[int], Sequence[float]]]:
        if top_k:
            return measure_top_probabilities(numpy.asarray(indices, dtype=numpy.int32), self.n, top_k, self.state)
        else:
            return measure_probabilities(numpy.asarray(indices, dtype=numpy.int32), self.n, self.state)

    def get_state_size(self) -> int:
        return len(self.state)

    def get_relative_range(self, start: int, end: int) -> Sequence[complex]:
        return self.state[start:end]

    def overwrite_relative_range(self, start: int, end: int, data: Sequence[complex]):
        self.state[start:end] = data

    def addto_relative_range(self, start: int, end: int, data: Sequence[complex]):
        self.state[start:end] += data
