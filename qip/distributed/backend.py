from qip.distributed.proto import *
from qip.distributed.formatsock import FormatSocket
from qip.distributed.proto.conversion import *
from qip.backend import StateType, InitialState
from qip.util import InitialState, IndexType, MatrixType
from qip.operators import CMat, SwapMat
from typing import Union, Sequence, Any, Tuple, Mapping, Callable, Optional, MutableSequence, cast
import numpy
import socket
import ssl


class DistributedBackend(StateType):
    def __init__(self, n, server_host: str = 'localhost', server_port: int = 1708):
        super().__init__(n, None)
        self.control_server_addr = (server_host, server_port)

        sock = socket.socket()
        sock.connect(self.control_server_addr)
        # First byte is whether to use ssl or not, all remaining communication is proto based.
        b = sock.recv(1)
        if b != b'\x00':
            sock = ssl.wrap_socket(sock)
        self.socket = FormatSocket(sock)

        # Introduce yourself.
        host_info = HostInformation()
        host_info.client_info.name = 'backend'
        self.socket.send(host_info.SerializeToString())

    @staticmethod
    def make_state(n: int, index_groups: Sequence[Sequence[int]], feed_list: Sequence[InitialState],
                   statetype: type = numpy.complex128) -> StateType:
        distbackend = DistributedBackend(n)

        setup_message = StateSetup()
        setup_message.n = n
        for index_group, initial_state in zip(index_groups, feed_list):
            pb_state = setup_message.states.add()
            pb_state.indices.CopyFrom(indices_to_pbindices(index_group))

            if type(initial_state) == int:
                pb_state.index = initial_state
            else:
                pb_state.vector.CopyFrom(vec_to_pbvec(initial_state))

        distbackend.socket.send(setup_message.SerializeToString())
        resp = StateHandle.FromString(distbackend.socket.recv())

        if resp.HasField('error_message'):
            raise Exception(resp.error_message)
        else:
            distbackend.state = resp.state_handle
            return distbackend

    def kronselect_dot(self, mats: Mapping[IndexType, MatrixType],
                       input_offset: int = 0, output_offset: int = 0) -> None:
        workerop = WorkerOperation()
        for indices in mats:
            mat = mats[indices]
            matop_to_pbmatop(indices, mat, workerop.kronprod.matrices.add())
        workerop.job_id = self.state

        self.socket.send(workerop.SerializeToString())
        conf = WorkerConfirm.FromString(self.socket.recv())

        if conf.HasField('error_message'):
            self.close()
            raise Exception(conf.error_message)
        elif conf.job_id != self.state:
            self.close()
            raise Exception("Server miscommunication: {} != {}".format(self.state, conf.job_id))

    def func_apply(self, reg1_indices: IndexType, reg2_indices: IndexType, func: Callable[[int], int],
                   input_offset: int = 0, output_offset: int = 0) -> None:
        raise NotImplemented()

    def measure(self, indices: Sequence[int], measured: Optional[int] = None,
                measured_prob: Optional[float] = None,
                input_offset: int = 0, output_offset: int = 0):
        workerop = WorkerOperation()
        workerop.job_id = self.state
        workerop.measure.reduce = False
        indices_to_pbindices(indices, workerop.measure.indices)

        self.socket.send(workerop.SerializeToString())
        conf = WorkerConfirm.FromString(self.socket.recv())

        if conf.HasField('error_message'):
            raise Exception(conf.error_message)
        else:
            return conf.measure_result.measured_bits, conf.measure_result.measured_prob

    def reduce_measure(self, indices: IndexType, measured: Optional[int] = None,
                       measured_prob: Optional[float] = None,
                       input_offset: int = 0, output_offset: int = 0) -> Tuple[int, float]:
        workerop = WorkerOperation()
        workerop.job_id = self.state
        workerop.measure.reduce = True
        indices_to_pbindices(indices, workerop.measure.indices)

        self.socket.send(workerop.SerializeToString())
        conf = WorkerConfirm.FromString(self.socket.recv())

        if conf.HasField('error_message'):
            raise Exception(conf.error_message)
        else:
            return conf.measure_result.measured_bits, conf.measure_result.measured_prob

    def soft_measure(self, indices: Sequence[int], measured: Optional[int] = None,
                     input_offset: int = 0):
        raise NotImplemented()

    def measure_probabilities(self, indices: IndexType, top_k: int = 1024) -> Tuple[Sequence[int], Sequence[float]]:
        if not top_k:
            raise NotImplemented("Distributed backends don't support full probability dumps")

        workerop = WorkerOperation()
        workerop.job_id = self.state
        workerop.measure.top_k = top_k
        indices_to_pbindices(indices, workerop.measure.indices)

        self.socket.send(workerop.SerializeToString())
        conf = WorkerConfirm.FromString(self.socket.recv())

        if conf.HasField('error_message'):
            raise Exception(conf.error_message)
        else:
            return list(conf.measure_result.top_k_indices.index), list(conf.measure_result.top_k_probs)

    def close(self):
        close_op = WorkerOperation()
        close_op.close = True
        self.socket.send(close_op.SerializeToString())
        self.socket.close()
