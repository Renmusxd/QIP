from qip.ext.kronprod import cdot_loop
from qip.ext.kronprod import measure
from qip.ext.kronprod import measure_probabilities
from qip.ext.func_apply import func_apply
from qip.util import kronselect_dot, gen_edit_indices, InitialState, IndexType, MatrixType
import numpy
from typing import Union, Sequence, Any, Tuple, Mapping, Callable, Optional


class StateType:
    def __init__(self, state):
        self.state = state

    def get_state(self):
        return self.state


class Backend:
    def __init__(self, n: int):
        self.n = n

    def make_state(self, *args, **kwargs) -> StateType:
        raise NotImplemented("make_state not implemented by base class")

    def kronselect_dot(self, mats: Mapping[IndexType, MatrixType], vec: StateType, n: int,
                       outputarray: StateType):
        raise NotImplemented("kronselect_dot not implemented by base class")

    def func_apply(self, reg1_indices: Sequence[int], reg2_indices: Sequence[int], func: Callable[[int],int],
                   vec: StateType, n: int, output: StateType):
        raise NotImplemented("func_apply not implemented by base class")

    def measure(self, indices: Sequence[int], n: int, vec: StateType, out: StateType, measured: Optional[int] = None,
                measured_prob: Optional[float] = None):
        raise NotImplemented("measure not implemented by base class")

    def measure_probabilities(self, indices: Sequence[int], n: int, vec: StateType):
        raise NotImplemented("measure_probabilities not implemented by base class")


class CythonBackend(Backend):
    def __init__(self, n):
        super().__init__(n)

    def make_state(self, index_groups: Sequence[Sequence[int]], feed_list: Mapping[Sequence[int], InitialState],
                   state: StateType = None, statetype: type = numpy.complex128) -> Tuple[StateType, StateType]:
        # TODO rewrite in state type agnostic form
        if state is None:
            state = numpy.zeros(2 ** self.n, dtype=statetype)

            if len(feed_list) == 0:
                state[0] = 1

            # Set all the entries in state to product of matrix entries
            for index, flips in gen_edit_indices(index_groups, self.n - 1):
                state[index] = 1.0
                for qindex, flip in enumerate(flips):
                    state[index] = state[index] * feed_list[qindex][flip]
        elif type(state) == int:
            stateint = state
            state = numpy.zeros(2 ** self.n, dtype=statetype)
            state[stateint] = 1.0
        elif len(state) != 2 ** self.n:
            raise ValueError("State size must be 2**n")
        elif type(state) == list:
            state = numpy.array(state)
        return state, state.copy()

    def kronselect_dot(self, mats: Mapping[IndexType, MatrixType], vec: StateType, n: int,
                       outputarray: StateType):
        return kronselect_dot(mats, vec, n, outputarray, dot_impl=cdot_loop)

    def func_apply(self, reg1_indices: Sequence[int], reg2_indices: Sequence[int], func: Callable[[int],int],
                   vec: StateType, n: int, output: StateType):
        return func_apply(reg1_indices, reg2_indices, func, vec, n, output)

    def measure(self, indices: Sequence[int], n: int, vec: StateType, out: StateType, measured: Optional[int] = None,
                measured_prob: Optional[float] = None):
        return measure(indices, n, vec, out, measured=measured, measured_prob=measured_prob)

    def measure_probabilities(self, indices: Sequence[int], n: int, vec: StateType):
        return measure_probabilities(indices, n, vec)