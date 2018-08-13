from qip.distributed.proto import *
from qip.distributed.formatsock import FormatSocket
from qip.distributed.proto.conversion import *
from qip.backend import Backend, StateType, InitialState
from qip.util import InitialState, IndexType, MatrixType
from qip.operators import CMat, SwapMat
from typing import Union, Sequence, Any, Tuple, Mapping, Callable, Optional, MutableSequence, cast
import numpy
import socket
import ssl


class DistributedBackend(Backend):
    def __init__(self, n, server_host: str = 'localhost', server_port: int = 1708):
        super().__init__(n)
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

    def make_state(self, index_groups: Sequence[Sequence[int]], feed_list: Sequence[InitialState],
                   state: InitialState = None, startindex: Optional[int] = None, endindex: Optional[int] = None,
                   statetype: type = numpy.complex128) -> StateType:
        if startindex is not None or endindex is not None:
            raise ValueError("Distributed backends don't yet support subsections")

        if state is not None:
            raise ValueError("Cannot feed full state to distributed backend, only individual qubit states.")

        setup_message = StateSetup()
        setup_message.n = self.n
        for index_group, initial_state in zip(index_groups, feed_list):
            pb_state = setup_message.states.add()
            pb_state.indices.CopyFrom(indices_to_pbindices(index_group))

            if type(initial_state) == int:
                pb_state.index = initial_state
            else:
                pb_state.vector.CopyFrom(vec_to_pbvec(initial_state))

        self.socket.send(setup_message.SerializeToString())
        resp = StateHandle.FromString(self.socket.recv())

        if resp.HasField('error_message'):
            raise Exception(resp.error_message)
        else:
            return StateType(resp.state_handle)

    def kronselect_dot(self, mats: Mapping[IndexType, MatrixType], state: StateType, n: int,
                       input_offset: int = 0, output_offset: int = 0) -> None:
        kronprodop = KronProd()
        for indices in mats:
            mat = mats[indices]
            matop_to_pbmatop(indices, mat, kronprodop.matrices.add())
        workerop = WorkerOperation()
        workerop.job_id = state.state
        workerop.kronprod.CopyFrom(kronprodop)
        self.socket.send(workerop.SerializeToString())
        conf = WorkerConfirm.FromString(self.socket.recv())
        if conf.HasField('error_message'):
            self.close()
            raise Exception(conf.error_message)
        elif conf.job_id != state.state:
            self.close()
            raise Exception("Server miscommunication: {} != {}".format(state.state, conf.job_id))

    def func_apply(self, reg1_indices: IndexType, reg2_indices: IndexType, func: Callable[[int], int],
                   state: StateType, n: int, input_offset: int = 0, output_offset: int = 0) -> None:
        raise NotImplemented()

    def measure(self, indices: IndexType, n: int, state: StateType, measured: Optional[int] = None,
                measured_prob: Optional[float] = None) -> Tuple[StateType, int]:
        raise NotImplemented()

    def measure_probabilities(self, indices: IndexType, n: int, state: StateType) -> Sequence[float]:
        raise NotImplemented()

    def close(self):
        self.socket.close()

