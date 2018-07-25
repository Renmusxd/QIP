from qip.ext.kronprod import cdot_loop
from qip.ext.kronprod import measure
from qip.ext.kronprod import measure_probabilities
from qip.ext.func_apply import func_apply
from qip.util import kronselect_dot
from qip.util import gen_edit_indices
from qip.util import flatten
import numpy


class Backend:
    def __init__(self, n):
        self.n = n

    def make_state(self, index_groups, feed_list, state=None, inputstart=0, outputstart=0,
                   statetype=numpy.complex128):
        raise NotImplemented("make_state not implemented by base class")

    def kronselect_dot(self, mats, vec, n, outputarray, inputstart=0, outputstart=0):
        raise NotImplemented("kronselect_dot not implemented by base class")

    def func_apply(self, reg1_indices, reg2_indices, func, vec, n, output, inputstart=0, outputstart=0):
        raise NotImplemented("func_apply not implemented by base class")

    def measure(self, indices, n, vec, out, measured=None, measured_prob=None, inputstart=0, outputstart=0):
        raise NotImplemented("measure not implemented by base class")

    def measure_probabilities(self, indices, n, vec, inputstart=0, outputstart=0):
        raise NotImplemented("measure_probabilities not implemented by base class")

    def close(self):
        pass


class CythonBackend(Backend):
    def __init__(self, n):
        super().__init__(n)

    def make_state(self, index_groups, feed_list, state=None, startindex=None, endindex=None,
                   statetype=numpy.complex128):
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
        return state, state.copy()

    def kronselect_dot(self, mats, vec, n, outputarray, inputstart=0, outputstart=0):
        return kronselect_dot(mats, vec, n, outputarray, dot_impl=cdot_loop,
                              input_offset=inputstart, output_offset=outputstart)

    def func_apply(self, reg1_indices, reg2_indices, func, vec, n, output, inputstart=0, outputstart=0):
        return func_apply(reg1_indices, reg2_indices, func, vec, n, output,
                          input_offset=inputstart, output_offset=outputstart, pregen=True)

    def measure(self, indices, n, vec, out, measured=None, measured_prob=None, inputstart=0, outputstart=0):
        # TODO input and output offsets
        return measure(indices, n, vec, out, measured=measured, measured_prob=measured_prob)

    def measure_probabilities(self, indices, n, vec, inputstart=0, outputstart=0):
        # TODO input and output offsets
        return measure_probabilities(indices, n, vec)
