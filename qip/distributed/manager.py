from qip.distributed.proto import *
from qip.distributed.proto.conversion import *
from qip.distributed import formatsock
from collections import OrderedDict
import itertools
import random
import socket
import ssl
from typing import Callable, Sequence, List, Tuple, Iterable, Mapping
from threading import Thread, Lock, Condition
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
                    returned_workers = pool.send_op(op)
                    if returned_workers:
                        self.logger("[*] Returning {} workers to pool after operation.".format(len(returned_workers)))
                        with self.worker_lock:
                            self.free_workers.extend(returned_workers)

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
            self.logger("[*] Socket closed.")

    def make_state(self, setup: StateSetup) -> StateHandle:
        n = setup.n
        with self.worker_lock:
            pool = WorkerPool(n, Manager.WORKER_N, logger=self.logger)
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


class InsufficientResources(ValueError):
    def __init__(self, msg):
        super(InsufficientResources, self).__init__(msg)


class WorkerPool:
    def __init__(self, n: int, worker_n: int, logger: Callable[[str], None] = print):
        self.n = n
        self.worker_n = worker_n
        self.unallocated_workers = []
        self.workers = []
        self.logger = logger

    def draw_workers(self, workers: Sequence['AnnotatedSocket']) -> Sequence['AnnotatedSocket']:
        # Assume all equal.
        # Required along one side of matrix is 2**n/2**m == 2**(n-m), so total is that squared == 2**2(n-m) (at least 1)
        required_workers = pow(2, 2 * max(self.n - self.worker_n, 0))
        if len(workers) < required_workers:
            raise InsufficientResources("Not enough workers: Has {} but needs {}".format(len(workers), required_workers))
        self.unallocated_workers, remaining_workers = list(workers[:required_workers]), workers[required_workers:]

        return remaining_workers

    def allocate_workers(self, setup: StateSetup, job_id: str):
        # Now assign ranges
        workern = min(self.worker_n, self.n)
        workerseps = [i << workern for i in range(self.n - workern + 2)]
        # list of (x, x + 2**workern)
        workerranges = list(zip(workerseps[:-1], workerseps[1:]))

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
        if op.HasField('kronprod'):
            self.send_kronprod(op)
            return []
        elif op.HasField('measure'):
            if op.measure.soft:
                raise NotImplemented("Need to implement soft measurement")
            else:
                return self.send_reduce_measure(op)

    def send_kronprod(self, op: WorkerOperation):
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
        syncop.sync.set_up_to = 0
        syncop.job_id = op.job_id  # Part of same job
        for _, _, worker in self.workers:
            worker.sock.send(syncop.SerializeToString())

        # Receive confirmation from all
        for _, _, worker in self.workers:
            conf = WorkerConfirm.FromString(worker.sock.recv())
            if conf.HasField('error_message'):
                # TODO handle errors
                raise Exception(conf.error_message)

    def get_workers_up_to_output(self, threshold: int) -> Mapping[Tuple[int, int], List['AnnotatedSocket']]:
        reduction_workers = OrderedDict()
        for inputs, outputs, worker in self.workers:
            if outputs[1] > threshold:
                continue
            if inputs not in reduction_workers:
                reduction_workers[inputs] = []
            reduction_workers[inputs].append(worker)
        return reduction_workers

    def send_reduce_measure(self, op: WorkerOperation):
        # Get relevant rows which need to output reductions.
        indices = pbindices_to_indices(op.measure.indices)
        threshold = 2**(self.n - len(indices))
        self.logger("[*] Measurement threshold is {}".format(threshold))
        reduction_workers = self.get_workers_up_to_output(threshold)

        # First get all the worker probs
        probop = WorkerOperation()
        probop.total_prob = True
        probop.job_id = op.job_id  # Part of same job
        inputs = list(sorted(reduction_workers.keys()))
        first_workers = [reduction_workers[input_tuple][0] for input_tuple in inputs]
        confs = self.map_workers_via_op(probop, first_workers, throw_errors=True)

        # Find the measured input values.
        r = random.random()
        measured_inputs = None
        for ins, conf in zip(inputs, confs):
            r -= conf.measure_result.measured_prob
            if r <= 0:
                measured_inputs = ins
                break

        # Ask appropriate worker to get a measurement.
        first_worker = reduction_workers[measured_inputs][0]
        get_bits = WorkerOperation()
        get_bits.CopyFrom(op)
        get_bits.measure.soft = True
        first_worker.sock.send(get_bits.SerializeToString())
        conf = WorkerConfirm.FromString(first_worker.sock.recv())
        measured_bits = conf.measure_result.measured_bits
        self.logger("[*] Measured bits are {}".format(measured_bits))
        # Measured_prob is at least what was measured for the first worker.
        measured_prob = conf.measure_result.measured_prob

        # Accumulate probability from each worker.
        get_bits.measure.measure_result.measured_bits = measured_bits
        measure_prob_workers = [reduction_workers[ins][0] for ins in inputs
                                if ins != measured_inputs]
        confs = self.map_workers_via_op(get_bits, measure_prob_workers, throw_errors=True)
        measured_prob += sum(conf.measure_result.measured_prob for conf in confs)
        self.logger("[*] Measured prob is {}".format(measured_prob))

        # Flatten worker dictionary.
        all_reduction_workers = [worker for workers in reduction_workers.values() for worker in workers]

        # We not have the total probability for the measured_bits, we can reduce.
        reduce_op = WorkerOperation()
        reduce_op.CopyFrom(op)
        reduce_op.measure.measure_result.measured_bits = measured_bits
        reduce_op.measure.measure_result.measured_prob = measured_prob
        reduce_op.measure.soft = False
        self.logger("[*] Performing reduction measurement")
        self.map_workers_via_op(reduce_op, all_reduction_workers, throw_errors=True)

        # Now sync to smaller worker set and remove unnecessary workers.
        syncop = WorkerOperation()
        syncop.job_id = op.job_id
        syncop.sync.set_up_to = threshold
        self.logger("[*] Performing synchronization")
        self.map_workers_via_op(syncop, all_reduction_workers, throw_errors=True)

        # Now find which workers need to be returned to the pool.
        remaining_workers = []
        done_workers = []
        close_op = WorkerOperation()
        close_op.job_id = op.job_id
        close_op.close = True
        self.logger("[*] Closing obsolete workers...")
        for inputs, outputs, worker in self.workers:
            if inputs[1] <= threshold and outputs[1] <= threshold:
                remaining_workers.append((inputs, outputs, worker))
            else:
                worker.sock.send(close_op.SerializeToString())
                done_workers.append(worker)
        self.logger("\tClosed {} workers".format(len(done_workers)))

        self.workers = remaining_workers
        return done_workers

    def send_to_each_of(self, op: WorkerOperation, workers: Iterable['AnnotatedSocket']):
        for worker in workers:
            worker.sock.send(op.SerializeToString())

    def receive_from_each_of(self, workers: Iterable['AnnotatedSocket'],
                             throw_errors: bool = True) -> List[WorkerConfirm]:
        confs = []
        for worker in workers:
            conf = WorkerConfirm.FromString(worker.sock.recv())
            confs.append(conf)
            if throw_errors and conf.HasField('error_message'):
                raise Exception(conf.error_message)
        return confs

    def map_workers_via_op(self, op: WorkerOperation, workers: Sequence['AnnotatedSocket'],
                           throw_errors: bool = True) -> Sequence['WorkerConfirm']:
        self.send_to_each_of(op, workers)
        return self.receive_from_each_of(workers, throw_errors=throw_errors)

    def close(self, job_id: str) -> Sequence['AnnotatedSocket']:
        self.logger("[*] Closing pool with {} workers".format(len(self.workers)))
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
    elif host_info.HasField('client_info'):
        server.logger("[+] Received connection:\n{}".format(host_info))
        server.manager.add_client(sock, host_info)
        server.manager.serve_client(sock)


if __name__ == "__main__":
    man = Manager()
    manserver = ManagerServer('0.0.0.0', 1708, man)
    manserver.run()
