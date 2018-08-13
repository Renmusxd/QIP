"""This serves as a main function to run a worker."""
from qip.distributed.proto import *
from qip.distributed.proto.conversion import *
from qip.distributed.worker.worker_backends import SocketServerBackend, SocketWorkerBackend
from qip.distributed.formatsock import FormatSocket
from qip.backend import CythonBackend
import socket
from threading import Thread, Lock
from typing import Mapping, Tuple, Callable
import ssl
import sys


class WorkerInstance:
    """A class created to serve a particular circuit + state."""
    def __init__(self, serverapi: SocketServerBackend, workerpool: 'WorkerPoolServer',
                 setup: WorkerSetup, logger: Callable[[str], None] = print):
        """
        Create the WorkerInstance
        :param serverapi: backend class to communicate with the host that processes are done.
        :param setup: Setup provided by manager
        """
        self.serverapi = serverapi
        self.n = setup.n
        self.job_id = setup.state_handle
        self.inputstartindex = setup.state_index_start
        self.inputendindex = setup.state_index_end
        self.outputstartindex = setup.output_index_start
        self.outputendindex = setup.output_index_end
        self.backend = CythonBackend(self.n)
        self.pool = workerpool
        self.logger = logger
        indexgroups = list(pbindex_to_index(state.indices) for state in setup.states)
        feedstates = [get_state_value(state) for state in setup.states]

        # Contact other workers and create a backend.


        self.state = self.backend.make_state(indexgroups, feedstates,
                                             inputstartindex=self.inputstartindex, inputendindex=self.inputendindex,
                                             outputstartindex=self.outputstartindex, outputendindex=self.outputendindex,
                                             statetype=pbstatetype_to_statetype(setup.statetype))

    def run(self):
        while True:
            self.logger("[*] Waiting for operation...")
            operation = self.serverapi.receive_operation()
            self.logger("[*] Performing operation:")
            self.logger(operation)
            if operation.HasField('close'):
                break
            elif operation.HasField('kronprod'):
                kronprod = operation.kronprod
                mats = {}
                for matop in kronprod.matrices:
                    indices, mat = pbmatop_to_matop(matop)
                    mats[indices] = mat

                self.backend.kronselect_dot(mats, self.state, self.n,
                                            input_offset=self.inputstartindex,
                                            output_offset=self.outputstartindex)
            elif operation.HasField('measure'):
                raise NotImplemented("Need to do measure sometime...")
            elif operation.HasField('sync'):
                # This logic assumes all workers given equal share, if ever changed then this must be fixed.
                if self.inputstartindex == self.outputstartindex and self.inputendindex == self.outputendindex:
                    # Output becomes input
                    self.state.swap()

                    # Receive output from everything which outputs to same region, add to current input
                    self.pool.receiveStateIncrementsFromAll(self.job_id, self.state,
                                                            self.outputstartindex, self.outputendindex)

                    # Send current input to everything which takes input from same region.
                    self.pool.sendStateToAll(self.job_id, self.state, self.inputstartindex, self.inputendindex)

                else:
                    # Swap input and output
                    self.state.swap()

                    # Send current output to worker along diagonal with in/out equal to our output
                    self.pool.sendStateToOne(self.job_id, self.state,
                                             self.outputstartindex, self.outputendindex,
                                             self.outputstartindex, self.outputendindex)

                    # Receive new input from worker with in/out equal to our input. Set to current input.
                    self.pool.receiveStateFromOne(self.job_id, self.state,
                                                  self.inputstartindex, self.inputendindex,
                                                  self.inputstartindex, self.inputendindex)

            else:
                raise NotImplemented("Unknown operation: {}".format(operation))

            self.logger("[+] Operation done!")
            self.logger("\tReporting done...")
            self.serverapi.report_done(operation.job_id)
            self.logger("[+] Reported operation complete.")
        self.pool.closeConnections(self.job_id)
        del self.state


def get_state_value(state: State) -> Union[numpy.ndarray, int]:
    if state.HasField('vector'):
        return pbvec_to_vec(state.vector)
    else:
        return state.index


class WorkerPoolServer(Thread):
    def __init__(self, hostname: str = 'localhost', port: int = 0, logger: Callable[[str], None] = print):
        """
        Create a server and pool object for finding connections to other workers.
        :param hostname: address to contact this worker
        :param port: port to bind to (0 default means choose any open).
        """
        super().__init__()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind(('0.0.0.0', port))
        self.addr = hostname
        self.port = self.sock.getsockname()[1]  # don't use port in case it was 0
        # map from (job_id, inputstart, inputend) to [worker_socket]
        self.inputrange_workers = {}
        # map from (job_id, outputstart, outputend) to [worker_socket]
        self.outputrange_workers = {}

        # Full range
        self.workers = {}

        self.logger = logger

        self.workerlock = Lock()

    def run(self):
        self.sock.listen(5)
        while True:
            sock, _ = self.sock.accept()
            sock = FormatSocket(sock)
            workersetup = WorkerPartner.FromString(sock.recv())

            fullkey = (workersetup.state_handle, workersetup.state_index_start, workersetup.state_index_end,
                       workersetup.output_index_start, workersetup.output_index_end)
            inputkey = (workersetup.state_handle, workersetup.state_index_start, workersetup.state_index_end)
            outputkey = (workersetup.state_handle, workersetup.output_index_start, workersetup.output_index_end)

            with self.workerlock:
                if inputkey not in self.inputrange_workers:
                    self.inputrange_workers[inputkey] = []
                if outputkey in self.outputrange_workers:
                    self.outputrange_workers[outputkey] = []

                self.workers[fullkey] = sock
                self.inputrange_workers[inputkey].append(sock)
                self.outputrange_workers[outputkey].append(sock)

    def sendStateToOne(self, job_id: str, state, inputstart: int, inputend: int, outputstart: int, outputend: int):
        sock = None
        sockkey = (job_id, inputstart, inputend, outputstart, outputend)
        with self.workerlock:
            if sockkey in self.workers:
                sock = self.workers[sockkey]
        if sock is not None:
            self.sendState(job_id, state, sock)

    def sendStateToAll(self, job_id: str, state, start: int, end: int, inputdefined=True):
        socks = []
        rangekey = (job_id, start, end)
        if inputdefined:
            with self.workerlock:
                if rangekey in self.inputrange_workers:
                    socks = self.inputrange_workers[rangekey]
        else:
            with self.workerlock:
                if rangekey in self.outputrange_workers:
                    socks = self.outputrange_workers[rangekey]
        for sock in socks:
            self.sendState(job_id, state, sock)

    def receiveStateFromOne(self, job_id: str, state, inputstart: int, inputend: int, outputstart: int, outputend: int):
        sock = None
        rangekey = (job_id, inputstart, inputend, outputstart, outputend)
        with self.workerlock:
            if rangekey in self.workers:
                sock = self.workers[rangekey]
        if sock is not None:
            self.receiveState(job_id, state, sock, overwrite=True)

    def receiveStateIncrementsFromAll(self, job_id: str, state, start: int, end: int, inputdefined=False):
        socks = []
        rangekey = (job_id, start, end)
        if inputdefined:
            with self.workerlock:
                if rangekey in self.inputrange_workers:
                    socks = self.inputrange_workers[rangekey]
        else:
            with self.workerlock:
                if rangekey in self.outputrange_workers:
                    socks = self.outputrange_workers[rangekey]
        for sock in socks:
            self.receiveState(job_id, state, sock, overwrite=False)

    # TODO rewrite as cython/c extension to improve speed.
    def sendState(self, job_id: str, state, sock: socket):
        syncaccept = SyncAccept.FromString(sock.recv())
        if syncaccept.job_id != job_id:
            raise Exception("Job ids do not match: {}/{}".format(job_id, syncaccept.job_id))
        inflight = 0

        # Break abstraction barrier now
        # TODO: unbreak it
        last_index = 0
        total_size = len(state.state)
        while last_index < total_size:
            while inflight < syncaccept.max_inflight and last_index < total_size:
                end_index = last_index + min(total_size - last_index, syncaccept.chunk_size)

                syncstate = SyncState()
                syncstate.job_id = job_id
                syncstate.rel_start_index = last_index
                vec_to_pbvec(state.state[last_index:end_index], pbvec=syncstate.data)

                inflight += 1
                last_index = end_index

                syncstate.done = last_index == total_size
                sock.send(syncstate.SerializeToString())

            # Overwrite syncaccept to allow live changes to inflight and chunk_size.
            syncaccept = SyncAccept.FromString(sock.recv())
            inflight -= 1
            if syncaccept.job_id != job_id:
                raise Exception("Job ids do not match: {}/{}".format(job_id, syncaccept.job_id))

    # TODO rewrite as cython/c extension to improve speed.
    def receiveState(self, job_id: str, state, sock: socket, overwrite=False, chunk_size=2048, max_inflight=4):
        syncaccept = SyncAccept()
        syncaccept.job_id = job_id
        syncaccept.chunk_size = chunk_size
        syncaccept.max_inflight = max_inflight
        # Let the other side know we are ready to receive and how to send info.
        sock.send(syncaccept.SerializeToString())

        running = True
        while running:
            syncstate = SyncState.FromString(sock.recv())
            relstart = syncstate.rel_start_index
            data = pbvec_to_vec(syncstate.data)

            # Break abstraction barrier now
            # TODO: unbreak it
            if overwrite:
                state.state[relstart:relstart+len(data)] = data
            else:
                state.state[relstart:relstart+len(data)] += data

            sock.send(syncaccept.SerializeToString())
            running = not syncstate.done

    def makeConnection(self, addr: str, port: int, job_id: str,
                       myinputstart: int, myinputend: int, myoutputstart: int, myoutputend: int,
                       theirinputstart: int, theirinputend: int, theiroutputstart: int, theiroutputend: int):
        wp = WorkerPartner()
        wp.job_id = job_id
        wp.state_index_start = myinputstart
        wp.state_index_end = myinputend
        wp.output_index_start = myoutputstart
        wp.output_index_end = myoutputend

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((addr, port))
        sock = FormatSocket(sock)
        sock.send(wp.SerializeToString())

        fullkey = (job_id, theirinputstart, theirinputend, theiroutputstart, theiroutputend)
        inputkey = (job_id, theirinputstart, theirinputend)
        outputkey = (job_id, theiroutputstart, theiroutputend)
        with self.workerlock:
            self.workers[fullkey] = sock

            if inputkey not in self.inputrange_workers:
                self.inputrange_workers[inputkey] = []
            self.inputrange_workers[inputkey].append(sock)

            if outputkey not in self.outputrange_workers:
                self.outputrange_workers[outputkey] = []
            self.outputrange_workers[outputkey].append(sock)

    def closeConnections(self, job_id: Optional[str] = None):
        with self.workerlock:
            for rangekey in self.workers:
                if job_id is None or job_id == rangekey[0]:
                    self.workers.pop(rangekey).close()

            for inputkey in self.inputrange_workers:
                if job_id is None or job_id == inputkey[0]:
                    self.inputrange_workers.pop(inputkey)

            for outputkey in self.outputrange_workers:
                if job_id is None or job_id == outputkey[0]:
                    self.outputrange_workers.pop(outputkey)


class WorkerRunner:
    def __init__(self, addr: str = 'localhost', port: int = 1708, logger: Callable[[str], None] = print):
        sock = socket.socket()
        sock.connect((addr, port))
        b = sock.recv(1)
        if b != b'\x00':
            sock = ssl.wrap_socket(sock)
        self.socket = FormatSocket(sock)
        workerinfo = HostInformation()
        workerinfo.worker_info.n_qubits = 28
        self.socket.send(workerinfo.SerializeToString())
        self.serverapi = SocketServerBackend(self.socket)
        self.addr = addr
        self.port = port

        self.pool = WorkerPoolServer(logger=logger)
        self.pool.start()

        self.logger = logger

    def run(self):
        self.logger("[*] Starting up...")
        while True:
            self.logger("[*] Waiting for setup...")
            setup = WorkerSetup.FromString(self.socket.recv())
            self.logger(setup)
            worker = WorkerInstance(self.serverapi, self.pool, setup)
            worker.run()


if __name__ == "__main__":
    host, port = sys.argv[1], int(sys.argv[2])
    wr = WorkerRunner(host, port)
    wr.run()
