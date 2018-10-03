"""This serves as a main function to run a worker."""
from qip.distributed.proto import *
from qip.distributed.proto.conversion import *
from qip.distributed.worker.worker_backends import SocketServerBackend, SocketWorkerBackend
from qip.distributed.worker.worker_logger import WorkerLogger, PrintLogger
from qip.distributed.formatsock import FormatSocket
from qip.backend import CythonBackend, StateType
import socket
from threading import Thread, Lock
from typing import Mapping, Tuple, Callable
import ssl
import sys


class WorkerInstance:
    """A class created to serve a particular circuit + state."""
    def __init__(self, serverapi: SocketServerBackend, workerpool: 'WorkerPoolServer',
                 setup: WorkerSetup, logger: WorkerLogger = None):
        """
        Create the WorkerInstance
        :param serverapi: backend class to communicate with the host that processes are done.
        :param setup: Setup provided by manager
        """
        if logger is None:
            self.logger = PrintLogger()
        else:
            self.logger = logger

        self.serverapi = serverapi
        self.n = setup.n
        self.job_id = setup.state_handle
        self.inputstartindex = setup.state_index_start
        self.inputendindex = setup.state_index_end
        self.outputstartindex = setup.output_index_start
        self.outputendindex = setup.output_index_end
        self.pool = workerpool
        indexgroups = list(pbindices_to_indices(state.indices) for state in setup.states)
        feedstates = [get_state_value(state) for state in setup.states]

        # Contact other workers and create a backend.
        # Should be changed if worker allocations change.
        if self.inputstartindex != self.outputstartindex:
            for partner in setup.partners:
                if partner.state_index_start != partner.output_index_start:
                    continue
                if partner.state_index_start == self.inputstartindex or partner.state_index_start == self.outputstartindex:
                    self.pool.make_connection(self.job_id, self.inputstartindex, self.inputendindex,
                                              self.outputstartindex, self.outputendindex, partner)

        self.logger.making_state(self.job_id, self.inputstartindex, self.inputendindex,
                                 self.outputstartindex, self.outputendindex)
        self.state = CythonBackend.make_state(self.n, indexgroups, feedstates,
                                              inputstartindex=self.inputstartindex, inputendindex=self.inputendindex,
                                              outputstartindex=self.outputstartindex, outputendindex=self.outputendindex,
                                              statetype=pbstatetype_to_statetype(setup.statetype))

    def run(self):
        while True:
            self.logger.waiting_for_operation(self.job_id)
            operation = self.serverapi.receive_operation()
            self.logger.running_operation(self.job_id, operation)

            self.logger(operation)
            if operation.HasField('close'):
                break

            elif operation.HasField('kronprod'):
                kronprod = operation.kronprod
                mats = {}
                for matop in kronprod.matrices:
                    indices, mat = pbmatop_to_matop(matop)
                    mats[indices] = mat
                self.state.kronselect_dot(mats, input_offset=self.inputstartindex, output_offset=self.outputstartindex)

            elif operation.HasField('total_prob'):
                # Probability when measuring all bits (0xFFFFFFFF)
                p = self.state.total_prob()
                self.logger("Reporting p={}".format(p))
                self.serverapi.report_probability(operation.job_id, measured_prob=p)
                continue

            elif operation.HasField('measure'):
                indices = pbindices_to_indices(operation.measure.indices)
                if operation.measure.soft:
                    measured = None
                    if operation.measure.HasField('measure_result'):
                        measured = operation.measure.measure_result.measured_bits
                    m, p = self.state.soft_measure(indices, measured=measured, input_offset=self.inputstartindex)
                else:
                    measured = None
                    measured_prob = None
                    if operation.measure.HasField('measure_result'):
                        measured = operation.measure.measure_result.measured_bits
                        measured_prob = operation.measure.measure_result.measured_prob
                    m, p = self.state.measure(indices, measured=measured, measured_prob=measured_prob,
                                              input_offset=self.inputstartindex)
                self.logger("Reporting m={}\tp={}".format(m,p))
                self.serverapi.report_probability(operation.job_id, measured_bits=m, measured_prob=p)
                continue

            elif operation.HasField('sync'):
                # This logic assumes all workers given equal share, if ever changed then this must be fixed.
                if self.inputstartindex == self.outputstartindex and self.inputendindex == self.outputendindex:
                    # Receive output from everything which outputs to same region, add to current input
                    self.logger.receiving_state(self.job_id)
                    self.pool.receive_state_increments_from_all(self.job_id, self.state,
                                                                self.outputstartindex, self.outputendindex)

                    # Send current input to everything which takes input from same region.
                    if operation.sync.HasField('set_up_to') and operation.sync.set_up_to:
                        self.logger.sending_state(self.job_id)
                        self.pool.send_state_up_to(self.job_id, self.state, self.inputstartindex, self.inputendindex,
                                                   operation.sync.set_up_to)
                    else:
                        self.logger.sending_state(self.job_id)
                        self.pool.send_state_to_all(self.job_id, self.state, self.inputstartindex, self.inputendindex)

                else:
                    # Swap input and output
                    # Send current output to worker along diagonal with in/out equal to our output
                    self.logger.sending_state(self.job_id)
                    self.pool.send_state_to_one(self.job_id, self.state,
                                                self.outputstartindex, self.outputendindex,
                                                self.outputstartindex, self.outputendindex)

                    should_receive = True
                    if operation.sync.HasField('set_up_to') and operation.sync.set_up_to:
                        should_receive = False
                        if operation.sync.set_up_to >= self.inputendindex and operation.sync.set_up_to >= self.outputendindex:
                            should_receive = True

                    if should_receive:
                        # Receive new input from worker with in/out equal to our input. Set to current input.
                        self.logger.receiving_state(self.job_id)
                        self.pool.receive_state_from_one(self.job_id, self.state,
                                                         self.inputstartindex, self.inputendindex,
                                                         self.inputstartindex, self.inputendindex)

            else:
                raise NotImplemented("Unknown operation: {}".format(operation))

            # If didn't override report system (see measurement), report done.
            self.logger.done_running_operation(self.job_id, operation)
            self.serverapi.report_done(operation.job_id)
        self.logger.closing_state(self.job_id)
        self.pool.close_connections(self.job_id)
        del self.state


def get_state_value(state: State) -> Union[numpy.ndarray, int]:
    if state.HasField('vector'):
        return pbvec_to_vec(state.vector)
    else:
        return state.index


class WorkerPoolServer(Thread):
    def __init__(self, hostname: str = '0.0.0.0', port: int = 0, logger: WorkerLogger = None):
        """
        Create a server and pool object for finding connections to other workers.
        :param hostname: address to contact this worker
        :param port: port to bind to (0 default means choose any open).
        """
        super().__init__()
        if logger is None:
            self.logger = PrintLogger()
        else:
            self.logger = logger

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind((hostname, port))
        self.addr = hostname
        self.port = self.sock.getsockname()[1]  # don't use port in case it was 0
        # map from (job_id, inputstart, inputend) to [worker_socket]
        self.inputrange_workers = {}
        # map from (job_id, outputstart, outputend) to [worker_socket]
        self.outputrange_workers = {}

        # Full range
        self.workers = {}

        self.workerlock = Lock()

    def run(self):
        self.logger.starting_server()
        self.sock.listen(5)
        self.logger.accepting_connections()
        while True:
            sock, _ = self.sock.accept()
            sock = FormatSocket(sock)
            workersetup = WorkerPartner.FromString(sock.recv())
            self.logger.accepted_connection()

            fullkey = (workersetup.job_id, workersetup.state_index_start, workersetup.state_index_end,
                       workersetup.output_index_start, workersetup.output_index_end)
            inputkey = (workersetup.job_id, workersetup.state_index_start, workersetup.state_index_end)
            outputkey = (workersetup.job_id, workersetup.output_index_start, workersetup.output_index_end)

            with self.workerlock:
                if inputkey not in self.inputrange_workers:
                    self.inputrange_workers[inputkey] = []
                if outputkey not in self.outputrange_workers:
                    self.outputrange_workers[outputkey] = []

                self.workers[fullkey] = sock
                self.inputrange_workers[inputkey].append((sock, workersetup.output_index_start, workersetup.output_index_end))
                self.outputrange_workers[outputkey].append((sock, workersetup.state_index_start, workersetup.state_index_end))

    def send_state_to_one(self, job_id: str, state: CythonBackend, inputstart: int, inputend: int,
                          outputstart: int, outputend: int, sock: Optional[FormatSocket] = None):
        sockkey = (job_id, inputstart, inputend, outputstart, outputend)
        if sock is None:
            with self.workerlock:
                if sockkey in self.workers:
                    sock = self.workers[sockkey]
        if sock is not None:
            self.logger("Sending state to {}".format(sockkey))
            self.send_state(job_id, state, sock)
        else:
            self.logger.log_error("Cannot send - No socket for {}".format(sockkey))

    def send_state_to_all(self, job_id: str, state: CythonBackend, start: int, end: int,
                          inputdefined=True):
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
        self.logger("Sending state to {} workers...".format(len(socks)))
        for i, (sock, _, _) in enumerate(socks):
            self.logger("\tSending to worker #{}".format(i))
            self.send_state(job_id, state, sock)

    def send_state_up_to(self, job_id: str, state: CythonBackend, inputstart: int, inputend: int, threshold: int = 0):
        rangekey = (job_id, inputstart, inputend)
        with self.workerlock:
            # We are either alone or broken.
            if rangekey not in self.inputrange_workers:
                return
            for sock, outputstart, outputend in self.inputrange_workers[rangekey]:
                if outputstart >= threshold:
                    continue
                self.send_state(job_id, state, sock)

    def receive_state_from_one(self, job_id: str, state: CythonBackend, inputstart: int, inputend: int,
                               outputstart: int, outputend: int):
        sock = None
        rangekey = (job_id, inputstart, inputend, outputstart, outputend)
        with self.workerlock:
            if rangekey in self.workers:
                sock = self.workers[rangekey]
        if sock is not None:
            self.logger("Receiving state from {}".format(rangekey))
            self.receive_state(job_id, state, sock, overwrite=True)
        else:
            self.logger.log_error("Cannot receive - No socket for {}".format(rangekey))

    def receive_state_increments_from_all(self, job_id: str, state: CythonBackend,
                                          start: int, end: int, inputdefined=False):
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
        self.logger("Receiving state from {} workers...".format(len(socks)))
        for sock, _, _ in socks:
            self.receive_state(job_id, state, sock, overwrite=False)

    # TODO rewrite as cython/c extension to improve speed.
    def send_state(self, job_id: str, state: StateType, sock: socket):
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
    def receive_state(self, job_id: str, state: CythonBackend, sock: socket, overwrite: bool = False,
                      chunk_size: int = 2048, max_inflight: int = 4):
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

    def make_connection(self, job_id: str, myinputstart: int, myinputend: int, myoutputstart: int, myoutputend: int,
                        partner: WorkerPartner):
        self.logger("Connecting to: {}".format(str(partner)))
        wp = WorkerPartner()
        wp.job_id = job_id
        wp.state_index_start = myinputstart
        wp.state_index_end = myinputend
        wp.output_index_start = myoutputstart
        wp.output_index_end = myoutputend

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((partner.addr, partner.port))
        sock = FormatSocket(sock)
        sock.send(wp.SerializeToString())

        fullkey = (job_id, partner.state_index_start, partner.state_index_end,
                   partner.output_index_start, partner.output_index_end)
        inputkey = (job_id, partner.state_index_start, partner.state_index_end)
        outputkey = (job_id, partner.output_index_start, partner.output_index_end)
        with self.workerlock:
            self.workers[fullkey] = sock

            if inputkey not in self.inputrange_workers:
                self.inputrange_workers[inputkey] = []
            self.inputrange_workers[inputkey].append((sock, partner.output_index_start, partner.output_index_end))

            if outputkey not in self.outputrange_workers:
                self.outputrange_workers[outputkey] = []
            self.outputrange_workers[outputkey].append((sock, partner.state_index_start, partner.state_index_end))

    def close_connections(self, job_id: Optional[str] = None):
        with self.workerlock:
            rangekeys = list(self.workers.keys())
            for rangekey in rangekeys:
                if job_id is None or job_id == rangekey[0]:
                    self.workers.pop(rangekey).close()

            inputkeys = list(self.inputrange_workers.keys())
            for inputkey in inputkeys:
                if job_id is None or job_id == inputkey[0]:
                    self.inputrange_workers.pop(inputkey)

            outputkeys = list(self.outputrange_workers.keys())
            for outputkey in outputkeys:
                if job_id is None or job_id == outputkey[0]:
                    self.outputrange_workers.pop(outputkey)


class WorkerRunner(Thread):
    def __init__(self, addr: str = 'localhost', port: int = 1708, bindaddr: str = 'localhost', bindport: int = 0,
                 logger: WorkerLogger = None):
        super().__init__()

        if logger is None:
            self.logger = PrintLogger()
        else:
            self.logger = logger

        sock = socket.socket()
        sock.connect((addr, port))
        b = sock.recv(1)
        if b != b'\x00':
            sock = ssl.wrap_socket(sock)
        self.socket = FormatSocket(sock)

        self.pool = WorkerPoolServer(bindaddr, bindport, logger=logger)
        self.pool.start()

        workerinfo = HostInformation()
        workerinfo.worker_info.n_qubits = 28
        workerinfo.worker_info.address = self.pool.addr
        workerinfo.worker_info.port = self.pool.port

        self.socket.send(workerinfo.SerializeToString())
        self.serverapi = SocketServerBackend(self.socket)
        self.addr = addr
        self.port = port
        self.running = True

    def run(self):

        while self.running:
            self.logger.waiting_for_setup()
            setup = WorkerSetup.FromString(self.socket.recv())
            self.logger.accepted_setup(setup)
            worker = WorkerInstance(self.serverapi, self.pool, setup, logger=self.logger)
            worker.run()


if __name__ == "__main__":
    host, port = 'localhost', 1708
    if len(sys.argv) > 1:
        host = sys.argv[1]
    if len(sys.argv) > 2:
        port = int(sys.argv[2])
    wr = WorkerRunner(host, port)
    wr.start()
    wr.join()
