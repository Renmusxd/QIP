from qip.backend import Backend
import numpy
from qip.distributed.messages import StateSetup


class DistributedBackend(Backend):
    def __init__(self, n):
        super().__init__(n)

    def make_state(self, index_groups, feed_list, state=None, startindex=None, endindex=None, statetype=numpy.complex128):
        pass

    def kronselect_dot(self, mats, vec, n, outputarray):
        raise NotImplemented("kronselect_dot not implemented by base class")

    def func_apply(self, reg1_indices, reg2_indices, func, vec, n, output):
        raise NotImplemented("func_apply not implemented by base class")

    def measure(self, indices, n, vec, out, measured=None, measured_prob=None):
        raise NotImplemented("measure not implemented by base class")

    def measure_probabilities(self, indices, n, vec):
        raise NotImplemented("measure_probabilities not implemented by base class")

    def close(self):
        # TODO Close connection to server.
        pass