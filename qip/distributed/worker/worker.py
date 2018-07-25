"""This serves as a main function to run a worker."""
from qip.distributed.messages import WorkerSetup, WorkerOperation
from qip.distributed.worker.worker_backends import SocketServerBackend, SocketWorkerBackend
from qip.distributed.formatsock import FormatSocket
from qip.backend import CythonBackend
import numpy
import socket
import ssl


class WorkerInstance:
    """A class created to serve a particular circuit + state."""
    def __init__(self, serverapi, workerapis, setup):
        """
        Create the WorkerInstance
        :param serverapi: backend class to communicate with the host that processes are done.
        :param workerapis: dictionary from index ranges (start, end) to backend classes
        :param startindex: first index represented by state
        :param endindex: index after last index represented by state (like range(start,end))
        """
        self.serverapi = serverapi
        self.workerapis = workerapis
        self.n = setup.n
        self.inputstartindex = setup.inputstartindex
        self.inputendindex = setup.inputendindex
        self.outputstartindex = setup.outputstartindex
        self.outputendindex = setup.outputendindex
        self.backend = CythonBackend(self.n)
        # Make input and output shape
        self.state = self.backend.make_state(setup.indexgroups, setup.feedstates,
                                             startindex=self.inputstartindex, endindex=self.inputendindex,
                                             statetype=setup.statetype)
        self.arena = numpy.zeros(shape=(self.outputendindex - self.outputstartindex,), dtype=self.state.dtype)

    def run(self):
        while True:
            operation = self.serverapi.receive_operation()
            if operation.opcommand == WorkerOperation.DONE:
                break

            if operation.opcommand == WorkerOperation.KRONPROD:
                self.backend.kronselect_dot(operation.mats, self.state, self.n, self.arena,
                                            inputstart=self.inputstartindex)
                self.state, self.arena = self.arena, self.state

            # TODO perform other operations
            self.serverapi.report_done()

    def clean(self):
        del self.state
        del self.arena


class WorkerRunner:
    def __init__(self, addr, port):
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
            setup = WorkerSetup.from_json(self.socket.recv())
            worker = WorkerInstance(self.serverapi, self.workers, setup)
            worker.run()
            worker.clean()