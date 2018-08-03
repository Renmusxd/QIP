from qip.ext.kronprod import cdot_loop
from qip.ext.kronprod import measure
from qip.ext.kronprod import measure_probabilities
from qip.ext.func_apply import func_apply
from qip.util import kronselect_dot, gen_edit_indices, InitialState, IndexType, MatrixType
import numpy
from typing import Union, Sequence, Any, Tuple, Mapping, Callable, Optional, MutableSequence, cast


class StateType:
    """
    A wrapper class which can contain either a state or a handle for a remote state.
    The state variable should only be accessed by subclasses of Backend which issues the StateType in the first place.
    Clients don't "know" what type StateType.state actually is.
    """
    def __init__(self, state):
        self.state = state


class Backend:
    def __init__(self, n: int):
        self.n = n

    def make_state(self, *args, **kwargs) -> StateType:
        raise NotImplemented("make_state not implemented by base class")

    def kronselect_dot(self, mats: Mapping[IndexType, MatrixType], vec: StateType, n: int,
                       outputarray: StateType) -> None:
        raise NotImplemented("kronselect_dot not implemented by base class")

    def func_apply(self, reg1_indices: Sequence[int], reg2_indices: Sequence[int], func: Callable[[int],int],
                   vec: StateType, n: int, output: StateType) -> None:
        raise NotImplemented("func_apply not implemented by base class")

    def measure(self, indices: Sequence[int], n: int, vec: StateType, out: StateType, measured: Optional[int] = None,
                measured_prob: Optional[float] = None) -> Tuple[StateType, StateType, int]:
        raise NotImplemented("measure not implemented by base class")

    def measure_probabilities(self, indices: Sequence[int], n: int, vec: StateType) -> Sequence[float]:
        raise NotImplemented("measure_probabilities not implemented by base class")

    def close(self):
        pass


class CythonBackend(Backend):
    def __init__(self, n: int):
        super().__init__(n)

    def make_state(self, index_groups: Sequence[Sequence[int]], feed_list: Mapping[Sequence[int], InitialState],
                   state: InitialState = None, startindex: Optional[int] = None, endindex: Optional[int] = None,
                   statetype: type = numpy.complex128) -> Tuple[StateType, StateType]:
        if state is None:
            if startindex is None:
                startindex = 0
            if endindex is None:
                endindex = 2**self.n

            state = numpy.zeros(endindex - startindex, dtype=statetype)

            if len(feed_list) == 0:
                state[0] = 1

            # Set all the entries in state to product of matrix entries
            for index, flips in gen_edit_indices(index_groups, self.n - 1):
                # Skip out of range
                if index < startindex or index >= endindex:
                    continue
                # Make index value
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
        return StateType(state), StateType(state.copy())

    def kronselect_dot(self, mats: Mapping[IndexType, MatrixType], vec: StateType, n: int,
                       outputarray: StateType) -> None:
        kronselect_dot(mats, vec.state, n, outputarray.state, dot_impl=cdot_loop)

    def func_apply(self, reg1_indices: Sequence[int], reg2_indices: Sequence[int], func: Callable[[int],int],
                   vec: StateType, n: int, output: StateType) -> None:
        func_apply(reg1_indices, reg2_indices, func, vec.state, n, output.state)

    def measure(self, indices: Sequence[int], n: int, vec: StateType, out: StateType, measured: Optional[int] = None,
                measured_prob: Optional[float] = None) -> Tuple[StateType, StateType, int]:
        inputvals = vec.state
        arena = out.state

        bits = measure(indices, n, inputvals, arena, measured=measured, measured_prob=measured_prob)

        # Cut and kill old memory after measurement so that footprint never grows above original.
        tmp_size = inputvals.shape[0]
        tmp_dtype = inputvals.dtype
        del inputvals

        new_inputvals = numpy.ndarray(shape=(tmp_size >> self.n), dtype=tmp_dtype)

        # Copy out relevant area from old arena
        new_arena = arena[:arena.shape[0] >> self.n]
        del arena

        return StateType(new_arena), StateType(new_inputvals), bits

    def measure_probabilities(self, indices: Sequence[int], n: int, vec: StateType) -> Sequence[float]:
        return measure_probabilities(indices, n, vec.state)
