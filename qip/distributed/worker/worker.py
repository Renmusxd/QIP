"""This serves as a main function to run a worker."""
from qip.distributed.proto import WorkerSetup, WorkerOperation
from qip.distributed.worker.worker_backends import SocketServerBackend, SocketWorkerBackend
from qip.distributed.formatsock import FormatSocket
from qip.backend import CythonBackend
import socket
from typing import Mapping, Tuple
import ssl


class WorkerInstance:
    """A class created to serve a particular circuit + state."""
    def __init__(self, serverapi: SocketServerBackend, workerapis: Mapping[Tuple[int, int], SocketWorkerBackend],
                 setup: WorkerSetup):
        """
        Create the WorkerInstance
        :param serverapi: backend class to communicate with the host that processes are done.
        :param workerapis: dictionary from index ranges (start, end) to backend classes
        :param setup: Setup provided by manager
        """
        self.serverapi = serverapi
        self.workerapis = workerapis
        self.n = setup.n
        self.inputstartindex = setup.state_index_start
        self.inputendindex = setup.state_index_end
        self.outputstartindex = setup.output_index_start
        self.outputendindex = setup.output_index_end
        self.backend = CythonBackend(self.n)
        # Make input and output shape
        self.state = self.backend.make_state(setup.indexgroups, setup.feedstates,
                                             startindex=self.inputstartindex, endindex=self.inputendindex,
                                             statetype=setup.statetype)

    def run(self):
        while True:
            operation = self.serverapi.receive_operation()
            if operation.opcommand == WorkerOperation.DONE:
                break

            if operation.opcommand == WorkerOperation.KRONPROD:
                self.backend.kronselect_dot(operation.mats, self.state, self.n,
                                            startindex=self.inputstartindex)

            # TODO perform other operations
            self.serverapi.report_done(operation.job_id)

    def clean(self):
        del self.state


class WorkerRunner:
    def __init__(self, addr: str = 'localhost', port: int = 1708):
        sock = socket.socket()
        sock.connect((addr, port))
        b = sock.recv(1)
        if b != b'\x00':
            sock = ssl.wrap_socket(sock)
        self.socket = FormatSocket(sock)
        self.serverapi = SocketServerBackend(self.socket)
        self.workers = {}
        self.addr = addr
        self.port = port

    def run(self):
        while True:
            setup = WorkerSetup.FromString(self.socket.recv())
            worker = WorkerInstance(self.serverapi, self.workers, setup)
            worker.run()
            worker.clean()
