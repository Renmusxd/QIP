from qip.distributed.proto import StateSetup, StateHandler
from qip.distributed.formatsock import FormatSocket
from qip.distributed.proto.conversion import *
from qip.backend import Backend, StateType, InitialState
from qip.util import InitialState, IndexType, MatrixType
from typing import Union, Sequence, Any, Tuple, Mapping, Callable, Optional, MutableSequence, cast
import numpy
import socket
import ssl


class DistributedBackend(Backend):
    def __init__(self, n, server_host: str, server_port: int = 1708):
        super().__init__(n)
        self.control_server_addr = (server_host, server_port)

        sock = socket.socket()
        sock.connect(self.control_server_addr)
        b = sock.recv(1)
        if b != b'\x00':
            sock = ssl.wrap_socket(sock)
        self.socket = FormatSocket(sock)

    def make_state(self, index_groups: Sequence[Sequence[int]], feed_list: Mapping[Sequence[int], InitialState],
                   state: InitialState = None, startindex: Optional[int] = None, endindex: Optional[int] = None,
                   statetype: type = numpy.complex128) -> StateType:
        if startindex is not None or endindex is not None:
            raise ValueError("Distributed backends don't yet support subsections")

        if state is not None:
            raise ValueError("Cannot feed full state to distributed backend, only individual qubit states.")

        setup_message = StateSetup()
        setup_message.n = self.n
        for index_group in feed_list:
            pb_state = setup_message.states.add()
            pb_state.indices = indices_to_pbindices(index_group)

            initial_state = feed_list[index_group]
            if type(initial_state) == int:
                pb_state.index = initial_state
            else:
                pb_state.vector = vec_to_pbvec(initial_state)

        self.socket.send(setup_message.SerializeToString())
        resp = StateHandler.FromString(self.socket.recv())

        if resp.has_error_message():
            raise Exception(resp.error_message)
        else:
            return StateType(resp.state_handle)

    def kronselect_dot(self, mats: Mapping[IndexType, MatrixType], state: StateType, n: int) -> None:
        raise NotImplemented()

    def func_apply(self, reg1_indices: Sequence[int], reg2_indices: Sequence[int], func: Callable[[int],int],
                   state: StateType, n: int) -> None:
        raise NotImplemented()

    def measure(self, indices: Sequence[int], n: int, state: StateType, measured: Optional[int] = None,
                measured_prob: Optional[float] = None) -> Tuple[StateType, int]:
        raise NotImplemented()

    def measure_probabilities(self, indices: Sequence[int], n: int, state: StateType) -> Sequence[float]:
        raise NotImplemented()

    def close(self):
        # TODO Close connection to server.
        pass