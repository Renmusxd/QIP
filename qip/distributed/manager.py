from qip.distributed.proto import WorkerSetup, HostInformation, StateSetup, StateHandle, WorkerOperation
from qip.distributed import formatsock
from threading import Thread, Lock, Condition
import socket
import ssl
from typing import Callable
import uuid


class Manager:
    INPUT_TIMEOUT = 1
    # TODO smarter allocation
    WORKER_N = 28

    def __init__(self, logger: Callable[[str], None] = print):
        self.logger = logger

        # map from state_handle to WorkerPool object
        self.job_pools = {}

        # list of workers who aren't in the job_pools
        # let's assume all workers are the same-ish for now.
        self.free_workers = []
        self.worker_lock = Lock()
        self.free_worker_con = Condition(self.worker_lock)

    def add_worker(self, workersock: socket, info: HostInformation):
        with self.worker_lock:
            self.free_workers.append(AnnotedSocket(workersock, info))
        t = Thread(target=self.serve_worker, args=(workersock, info))
        t.start()

    def add_client(self, clientsock: socket, info: HostInformation):
        t = Thread(target=self.serve_worker, args=(clientsock, info))
        t.start()

    def serve_client(self, sock: socket):
        # Wait for desired state
        setup = StateSetup.FromString(sock.recv())
        state_handle = self.make_state(setup)
        sock.send(state_handle.SerializeToString())

        # Apply operations
        running_job = True
        while running_job:
            op = WorkerOperation.FromString(sock.recv())
            with self.worker_lock:
                self.job_pools[state_handle.uuid_handle].send_op(op)
            running_job = not op.done
        sock.close()

    def make_state(self, setup: StateSetup) -> StateHandle:
        n = setup.n
        with self.worker_lock:
            sqr_diff = max(n - Manager.WORKER_N, 0)
            # Add the even/odd value to make it even, this makes it a perfect square, good for allocation
            num_workers = pow(2, sqr_diff + (sqr_diff % 2))

            if num_workers > len(self.free_workers):
                num_workers = len(self.free_workers)

            pool = WorkerPool(self.free_workers[:num_workers])
            uuid_handle = str(uuid.uuid4())
            self.job_pools[uuid_handle] = pool

            handle = StateHandle()
            handle.state_handle = uuid_handle
            return handle

    def serve_worker(self, sock: socket, info: HostInformation):
        # TODO
        pass


class WorkerPool:
    def __init__(self, workers):
        self.workers = workers
        self.allocate_workers()

    def allocate_workers(self):
        # TODO
        pass

    def send_op(self, op: WorkerOperation):
        # TODO
        pass


class AnnotedSocket:
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
        while True:
            self.logger("[*] Starting up...")
            clientsock, (ip, port) = self.tcpsock.accept()
            clientsock.settimeout(self.timeout_duration)
            try:
                self.logger("[*] Accepting connection")
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
                self.logger('\tWaiting for client information...')
                host_info = HostInformation.FromString(clientformatsock.recv())
                host_info.address = ip
                host_info.port = port

                if host_info.host_type == HostInformation.WORKER:
                    self.logger("[+] Received worker: {}".format(host_info))
                    self.manager.add_worker(clientsock, host_info)
                elif host_info.host_type == HostInformation.CLIENT:
                    self.logger("[+] Received connection: {}".format(host_info))
                    self.manager.add_client(clientformatsock, host_info)
            except IOError as e:
                self.logger("[!] Error accepting connection: {}".format(str(e)))
                try:
                    clientsock.close()
                except IOError as e:
                    self.logger("\tError while closing socket: {}".format(str(e)))


if __name__ == "__main__":
    man = Manager()
    server = ManagerServer('0.0.0.0', 1708, man)
    server.run()
