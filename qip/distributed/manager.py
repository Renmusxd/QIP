from qip.distributed.proto import *
from qip.distributed import formatsock
from threading import Thread, Lock, Condition
import socket
import ssl
from typing import Callable, Sequence, List
import uuid


class Manager:
    INPUT_TIMEOUT = 1
    # TODO smarter allocation
    WORKER_N = 9

    def __init__(self, logger: Callable[[str], None] = print):
        self.logger = logger

        # map from state_handle to WorkerPool object
        self.job_pools = {}

        # list of workers who aren't in the job_pools
        # let's assume all workers are the same-ish for now.
        self.free_workers = []
        self.worker_lock = Lock()
        self.pool_lock = Lock()

    def add_worker(self, workersock: socket, info: HostInformation):
        with self.worker_lock:
            self.free_workers.append(AnnotatedSocket(workersock, info))

    def add_client(self, clientsock: socket, info: HostInformation):
        pass

    def serve_client(self, sock: socket):
        # Wait for desired state
        self.logger("[*] Waiting for setup...")
        setup = StateSetup.FromString(sock.recv())
        self.logger("[*] Making state...")
        state_handle = self.make_state(setup)
        self.logger("\tMade state, sending handle: {}".format(state_handle.state_handle))
        sock.send(state_handle.SerializeToString())

        # Apply operations
        running_job = True
        try:
            while running_job:
                self.logger("[*] Waiting for operation for job: {}".format(state_handle.state_handle))
                op = WorkerOperation.FromString(sock.recv())
                if op.HasField("close"):
                    running_job = False
                else:
                    with self.pool_lock:
                        pool = self.job_pools[state_handle.state_handle]
                    pool.send_op(op)

                self.logger("[*] Sending confirmation to client...")
                conf = WorkerConfirm()
                conf.job_id = state_handle.state_handle
                sock.send(conf.SerializeToString())
        except Exception as e:
            self.logger("[!] Unknown exception: {}".format(e))
            conf = WorkerConfirm()
            conf.error_message = str(e)
            try:
                sock.send(conf.SerializeToString())
            except IOError as e:
                self.logger("\tError sending error code: {}".format(e))
        finally:
            with self.pool_lock:
                pool = self.job_pools[state_handle.state_handle]
            returned_workers = pool.close(state_handle.state_handle)

            self.logger("[*] Returning {} workers to pool.".format(len(returned_workers)))
            with self.worker_lock:
                self.free_workers.extend(returned_workers)
            sock.close()

    def make_state(self, setup: StateSetup) -> StateHandle:
        n = setup.n
        with self.worker_lock:
            pool = WorkerPool(n, Manager.WORKER_N)
            self.free_workers = pool.draw_workers(self.free_workers)
        state_handle = StateHandle()
        try:
            uuid_handle = str(uuid.uuid4())
            pool.allocate_workers(setup, uuid_handle)
            with self.pool_lock:
                self.job_pools[uuid_handle] = pool
            state_handle.state_handle = uuid_handle
            return state_handle
        except ValueError as e:
            state_handle.error_message = str(e)
            return state_handle

    def serve_worker(self, sock: socket, info: HostInformation):
        # TODO
        pass


class InsufficientResources(ValueError):
    def __init__(self, msg):
        super(InsufficientResources).__init__(self, msg)


class WorkerPool:
    def __init__(self, n: int, worker_n: int):
        self.n = n
        self.worker_n = worker_n
        self.unallocated_workers = []
        self.workers = []

    def draw_workers(self, workers: Sequence['AnnotatedSocket']) -> Sequence['AnnotatedSocket']:
        # Assume all equal.
        # Required along one side of matrix is 2**n/2**m == 2**(n-m), so total is that squared == 2**2(n-m) (at least 1)
        required_workers = pow(2, 2 * max(self.n - self.worker_n, 0))
        if len(workers) < required_workers:
            raise InsufficientResources("Not enough workers")
        self.unallocated_workers, remaining_workers = list(workers[:required_workers]), workers[required_workers:]

        return remaining_workers

    def allocate_workers(self, setup: StateSetup, job_id: str):
        # Now assign ranges
        workern = min(self.worker_n, self.n)
        workerseps = [i << workern for i in range(self.n - workern + 2)]
        # list of (x, x + 2**workern)
        workerranges = list(zip(workerseps[:-1], workerseps[1:]))

        print("Making ranges: {}".format(workerseps))
        for inputrange in workerranges:
            for outputrange in workerranges:
                self.workers.append((inputrange, outputrange, self.unallocated_workers.pop()))

        workersetup = WorkerSetup()
        workersetup.n = self.n
        workersetup.state_handle = job_id
        workersetup.states.extend(setup.states)
        for (inputstart, inputend), (outputstart, outputend), worker in self.workers:
            wp = workersetup.partners.add()
            wp.job_id = job_id
            wp.addr = worker.info.worker_info.address
            wp.port = worker.info.worker_info.port
            wp.state_index_start = inputstart
            wp.state_index_end = inputend
            wp.output_index_start = outputstart
            wp.output_index_end = outputend

        for (inputstart, inputend), (outputstart, outputend), worker in self.workers:
            workersetup.state_index_start = inputstart
            workersetup.state_index_end = inputend
            workersetup.output_index_start = outputstart
            workersetup.output_index_end = outputend
            worker.sock.send(workersetup.SerializeToString())

    def send_op(self, op: WorkerOperation):
        # Send to all
        for _, _, worker in self.workers:
            worker.sock.send(op.SerializeToString())
        # Receive confirmation from all
        for _, _, worker in self.workers:
            conf = WorkerConfirm.FromString(worker.sock.recv())
            if conf.HasField('error_message'):
                # TODO handle errors
                raise Exception(conf.error_message)

        # Tell all to sync.
        syncop = WorkerOperation()
        syncop.sync = True
        syncop.job_id = op.job_id  # Part of same job
        for _, _, worker in self.workers:
            worker.sock.send(syncop.SerializeToString())

        # Receive confirmation from all
        for _, _, worker in self.workers:
            conf = WorkerConfirm.FromString(worker.sock.recv())
            if conf.HasField('error_message'):
                # TODO handle errors
                raise Exception(conf.error_message)

    def close(self, job_id: str) -> Sequence['AnnotatedSocket']:
        close_op = WorkerOperation()
        close_op.job_id = job_id
        close_op.close = True
        for _, _, worker in self.workers:
            worker.sock.send(close_op.SerializeToString())
        return [w for _, _, w in self.workers]


class AnnotatedSocket:
    def __init__(self, sock, info):
        self.sock = sock
        self.info = info

    def fileno(self):
        return self.sock.fileno()


class ManagerServer:
    def __init__(self, host: str, port: int, manager: Manager,
                 certfile: str = None, keyfile: str = None, logger: Callable[[str], None] = print):
        self.manager = manager

        self.tcpsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcpsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.tcpsock.bind((host, port))

        self.ssl = certfile and keyfile
        self.certfile = certfile
        self.keyfile = keyfile

        self.timeout_duration = 1
        self.logger = logger

    def run(self):
        self.tcpsock.listen(5)

        self.logger("[*] Starting up...")
        while True:
            self.logger("[*] Accepting connections...")
            clientsock, (ip, port) = self.tcpsock.accept()
            clientsock.settimeout(self.timeout_duration)
            try:
                self.logger("[+] Accepted connection!")
                self.logger("\tSending SSL: " + ('ON' if self.ssl else 'OFF'))
                clientsock.send(b'\x01' if self.ssl else b'\x00')
                if self.ssl:
                    self.logger("\tWrapping socket...")
                    clientsock = ssl.wrap_socket(
                        clientsock,
                        server_side=True,
                        certfile=self.certfile,
                        keyfile=self.keyfile
                    )
                clientformatsock = formatsock.FormatSocket(clientsock)
                clientformatsock.settimeout(None)
                self.logger('\tWaiting for client information...')
                host_info = HostInformation.FromString(clientformatsock.recv())
                host_info.address = ip
                host_info.port = port
                t = Thread(target=handle, args=(self, clientformatsock, host_info))
                t.start()
            except IOError as e:
                self.logger("[!] Error accepting connection: {}".format(str(e)))
                try:
                    clientsock.close()
                except IOError as e:
                    self.logger("\tError while closing socket: {}".format(str(e)))


def handle(server: ManagerServer, sock: socket, host_info: HostInformation):
    if host_info.HasField('worker_info'):
        server.logger("[+] Received worker:\n{}".format(host_info))
        server.manager.add_worker(sock, host_info)
        server.manager.serve_worker(sock, host_info)
    elif host_info.HasField('client_info'):
        server.logger("[+] Received connection:\n{}".format(host_info))
        server.manager.add_client(sock, host_info)
        server.manager.serve_client(sock)


if __name__ == "__main__":
    man = Manager()
    manserver = ManagerServer('0.0.0.0', 1708, man)
    manserver.run()
