from qip.distributed.proto.conversion import *
from qip.distributed import formatsock
from qip.distributed.manager_logger import ServerLogger, PrintLogger
from collections import OrderedDict
import random
import socket
import ssl
from typing import Callable, Sequence, List, Tuple, Iterable, Mapping, Any
from threading import Thread, Lock, Condition
import uuid


class Manager:
    INPUT_TIMEOUT = 1

    def __init__(self, logger: ServerLogger = None):
        if logger is None:
            self.logger = PrintLogger()
        else:
            self.logger = logger

        # map from state_handle to WorkerPool object
        self.job_pools = {}

        # list of workers who aren't in the job_pools
        # let's assume all workers are the same for now.
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
        self.logger.waiting_for_setup()
        setup = StateSetup.FromString(sock.recv())
        state_handle = self.make_state(setup)
        self.logger.making_state(state_handle.state_handle, setup.n)
        sock.send(state_handle.SerializeToString())

        # Apply operations
        running_job = True
        try:
            while running_job:
                self.logger.waiting_for_operation(state_handle.state_handle)
                op = WorkerOperation.FromString(sock.recv())
                self.logger.running_operation(state_handle.state_handle, op)

                returned_data = None
                if op.HasField("close"):
                    running_job = False
                else:
                    with self.pool_lock:
                        pool = self.job_pools[state_handle.state_handle]
                    returned_workers, returned_data = pool.send_op(op)
                    if returned_workers:
                        self.logger.returning_workers(state_handle.state_handle, len(returned_workers))
                        with self.worker_lock:
                            self.free_workers.extend(returned_workers)

                self.logger.done_running_operation(state_handle.state_handle, op)
                conf = WorkerConfirm()
                conf.job_id = state_handle.state_handle

                if returned_data:
                    conf.measure_result.CopyFrom(returned_data)

                sock.send(conf.SerializeToString())
        except IOError as e:
            self.logger.log_error("Unknown exception: {}".format(e))
            conf = WorkerConfirm()
            conf.error_message = str(e)
            try:
                sock.send(conf.SerializeToString())
            except IOError as e:
                self.logger.log_error("Error sending error code: {}".format(e))
        finally:
            with self.pool_lock:
                pool = self.job_pools[state_handle.state_handle]
            returned_workers = pool.close(state_handle.state_handle)

            self.logger.returning_workers(state_handle.state_handle, len(returned_workers))
            with self.worker_lock:
                self.free_workers.extend(returned_workers)

            self.logger.closing_state(state_handle.state_handle)
            sock.close()

    def make_state(self, setup: StateSetup) -> StateHandle:
        n = setup.n
        with self.worker_lock:
            pool = WorkerPool(n, logger=self.logger)
            # TODO catch exception return error to client.
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

    def close(self):
        self.logger("Closing...")
        with self.worker_lock:
            with self.pool_lock:
                for job_id, pool in self.job_pools.items():
                    self.free_workers.extend(pool.close(job_id))
            for worker in self.free_workers:
                worker.sock.close()
        self.logger("Closed!")


class InsufficientResources(ValueError):
    def __init__(self, msg):
        super(InsufficientResources, self).__init__(msg)


class WorkerPool:
    def __init__(self, n: int, logger: ServerLogger):
        self.n = n
        self.unallocated_workers = []
        self.workers = []
        self.logger = logger
        self.min_worker_n = 0

    def draw_workers(self, workers: Sequence['AnnotatedSocket']) -> Sequence['AnnotatedSocket']:
        # Sort by n_qubits and draw from back using pop()
        copy_workers = list(sorted((w for w in workers), key=lambda w: w.info.worker_info.n_qubits))
        self.unallocated_workers = [copy_workers.pop()]
        worker_n = self.unallocated_workers[0].info.worker_info.n_qubits

        # Assume all equal.
        # Required along one side of matrix is 2**n/2**m == 2**(n-m), so total is that squared == 2**2(n-m) (at least 1)
        required_workers = pow(2, 2 * max(self.n - worker_n, 0))
        while len(self.unallocated_workers) < required_workers and len(copy_workers) > 0:
            to_pop = required_workers - len(self.unallocated_workers)
            for _ in range(to_pop):
                self.unallocated_workers.append(copy_workers.pop())
            # Get the new lowest worker_n and the required number of workers at that n.
            worker_n = self.unallocated_workers[-1].info.worker_info.n_qubits
            required_workers = pow(2, 2 * max(self.n - worker_n, 0))

        if len(workers) < required_workers:
            raise InsufficientResources("Not enough workers: Has {} but needs {}".format(len(workers), required_workers))
        self.min_worker_n = worker_n
        return copy_workers

    def allocate_workers(self, setup: StateSetup, job_id: str):
        # Now assign ranges
        workern = min(self.min_worker_n, self.n)

        workerseps = [i << workern for i in range(pow(2, self.n - workern) + 1)]
        # list of (x, x + 2**workern)
        workerranges = list(zip(workerseps[:-1], workerseps[1:]))

        for inputrange in workerranges:
            for outputrange in workerranges:
                self.workers.append((inputrange, outputrange, self.unallocated_workers.pop()))

        # Make sure all workers were allocated.
        assert len(self.unallocated_workers) == 0

        workercmd = WorkerCommand()
        workersetup = workercmd.setup
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
            worker.sock.send(workercmd.SerializeToString())

    def send_op(self, op: WorkerOperation) -> Tuple[Sequence['AnnotatedSocket'], Optional[MeasureResult]]:
        if op.HasField('kronprod'):
            self.send_kronprod(op)
            return [], None
        elif op.HasField('measure'):
            if op.measure.soft:
                raise NotImplemented("Need to implement soft measurement")
            elif op.measure.reduce:
                return self.send_reduce_measure(op)
            elif op.measure.top_k:
                if op.measure.top_k > 2048:
                    pass
                return [], self.send_measure_top(op)
            else:
                return [], self.send_measure(op)

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
            if outputs[0] > threshold:
                continue
            if inputs not in reduction_workers:
                reduction_workers[inputs] = []
            reduction_workers[inputs].append(worker)
        return reduction_workers

    def send_measure(self, op: WorkerOperation) -> MeasureResult:
        diagonal_workers = list(sorted((inputs, outputs, workers) for inputs, outputs, workers in self.workers
                                       if inputs == outputs))
        diagonal_worker_objs = [w for _, _, w in diagonal_workers]
        # First get all the worker probs
        probop = WorkerOperation()
        probop.total_prob = True
        probop.job_id = op.job_id  # Part of same job
        confs = self.map_workers_via_op(probop, diagonal_worker_objs, throw_errors=True)

        r = random.random()
        selected_worker = None
        i = 0
        print(r)
        for i, conf in enumerate(confs):
            print(conf)
            r -= conf.measure_result.measured_prob
            if r <= 0:
                _, _, selected_worker = diagonal_workers[i]
                break
        get_bits = WorkerOperation()
        get_bits.CopyFrom(op)
        get_bits.measure.soft = True
        get_bits.measure.reduce = False
        selected_worker.sock.send(get_bits.SerializeToString())
        conf = WorkerConfirm.FromString(selected_worker.sock.recv())
        measured_bits = conf.measure_result.measured_bits
        measured_prob = conf.measure_result.measured_prob
        self.logger("Measured bits are {}".format(measured_bits))

        # Accumulate probability from each worker (except the one that was already measured).
        get_bits.measure.measure_result.measured_bits = measured_bits
        noprob_workers = diagonal_worker_objs[:i] + diagonal_worker_objs[i+1:]
        confs = self.map_workers_via_op(get_bits, noprob_workers,
                                        throw_errors=True)
        measured_prob += sum(conf.measure_result.measured_prob for conf in confs)
        self.logger("Measured prob is {}".format(measured_prob))

        # We only need to ask the diagonals to perform the measurement operation
        measure_op = WorkerOperation()
        measure_op.CopyFrom(op)
        measure_op.measure.measure_result.measured_bits = measured_bits
        measure_op.measure.measure_result.measured_prob = measured_prob
        measure_op.measure.soft = False
        measure_op.measure.reduce = False
        self.logger("Performing measurement operation")
        self.map_workers_via_op(measure_op, diagonal_worker_objs, throw_errors=True)

        # Now sync all workers, but the non-diagonals don't need to send any state, only receive.
        syncop = WorkerOperation()
        syncop.job_id = op.job_id
        syncop.sync.diagonal_overwrite = True
        self.logger("Performing synchronization")
        self.map_workers_via_op(syncop, [w for _, _, w in self.workers], throw_errors=True)

        measure_res = MeasureResult()
        measure_res.measured_bits = measured_bits
        measure_res.measured_prob = measured_prob
        return measure_res

    def send_measure_top(self, op: WorkerOperation) -> MeasureResult:
        diagonal_workers = list(sorted((inputs, outputs, workers) for inputs, outputs, workers in self.workers
                                       if inputs == outputs))
        diagonal_worker_objs = [w for _, _, w in diagonal_workers]

        # Ask workers for top_k from each
        confs = self.map_workers_via_op(op, diagonal_worker_objs, throw_errors=True)

        print(confs)

        # Get minimum probability in each worker (for error measurement).
        conf_mins = [min(conf.measure_result.top_k_probs) for conf in confs]

        # Get each tuple sorted by index for aggregation. Reverse order speeds up pop()
        conf_results = [list(sorted(zip(conf.measure_result.top_k_indices.index, conf.measure_result.top_k_probs),
                                    reverse=True))
                        for conf in confs]
        total_results = sum(len(res) for res in conf_results)

        agg_top = []
        while total_results > 0:
            # Choose largest possible index as minimum
            min_index = min(res[-1][0] for res in conf_results)

            agg_p = 0
            agg_err = 0
            for i, res in enumerate(conf_results):
                # If the index appears for a given worker, add it's prob
                if res and res[-1][0] == min_index:
                    _, p = res.pop()
                    agg_p += p
                    total_results -= 1

                # Otherwise add the minimum prob returns by that worker to the possible error.
                else:
                    agg_err += conf_mins[i]
            agg_top.append((min_index, agg_p, agg_err))

        # Now sort by decreasing probability
        top_k_items = sorted(agg_top, key=lambda t: t[1], reverse=True)
        top_indices, top_probs, top_errors = zip(*top_k_items)

        measure_res = MeasureResult()
        measure_res.top_k_indices.index.extend(top_indices[:op.measure.top_k])
        measure_res.top_k_probs.extend(top_probs[:op.measure.top_k])
        measure_res.top_k_errors.extend(top_errors[:op.measure.top_k])
        return measure_res

    def send_reduce_measure(self, op: WorkerOperation) -> Tuple[Sequence['AnnotatedSocket'], MeasureResult]:
        # Get relevant rows which need to output reductions.
        indices = pbindices_to_indices(op.measure.indices)
        threshold = pow(2, self.n - len(indices))

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
        get_bits.measure.reduce = False
        first_worker.sock.send(get_bits.SerializeToString())
        conf = WorkerConfirm.FromString(first_worker.sock.recv())
        measured_bits = conf.measure_result.measured_bits
        self.logger("Measured bits are {}".format(measured_bits))
        # Measured_prob is at least what was measured for the first worker.
        measured_prob = conf.measure_result.measured_prob

        # Accumulate probability from each worker.
        get_bits.measure.measure_result.measured_bits = measured_bits
        measure_prob_workers = [reduction_workers[ins][0] for ins in inputs
                                if ins != measured_inputs]
        confs = self.map_workers_via_op(get_bits, measure_prob_workers, throw_errors=True)
        measured_prob += sum(conf.measure_result.measured_prob for conf in confs)
        self.logger("Measured prob is {}".format(measured_prob))

        # Flatten worker dictionary.
        all_reduction_workers = [worker for workers in reduction_workers.values() for worker in workers]

        # We not have the total probability for the measured_bits, we can reduce.
        reduce_op = WorkerOperation()
        reduce_op.CopyFrom(op)
        reduce_op.measure.measure_result.measured_bits = measured_bits
        reduce_op.measure.measure_result.measured_prob = measured_prob
        reduce_op.measure.soft = False
        reduce_op.measure.reduce = True
        self.logger("Performing reduction measurement")
        self.map_workers_via_op(reduce_op, all_reduction_workers, throw_errors=True)

        # Now sync to smaller worker set and remove unnecessary workers.
        syncop = WorkerOperation()
        syncop.job_id = op.job_id
        syncop.sync.set_up_to = threshold
        self.logger("Performing synchronization")
        self.map_workers_via_op(syncop, all_reduction_workers, throw_errors=True)

        # Now find which workers need to be returned to the pool.
        remaining_workers = []
        done_workers = []
        close_op = WorkerOperation()
        close_op.job_id = op.job_id
        close_op.close = True
        self.logger("Closing obsolete workers...")
        for inputs, outputs, worker in self.workers:
            if inputs[1] <= threshold and outputs[1] <= threshold:
                remaining_workers.append((inputs, outputs, worker))
            else:
                worker.sock.send(close_op.SerializeToString())
                done_workers.append(worker)
        self.logger("Closed {} workers".format(len(done_workers)))

        self.workers = remaining_workers

        measure_res = MeasureResult()
        measure_res.measured_bits = measured_bits
        measure_res.measured_prob = measured_prob
        return done_workers, measure_res

    def send_to_each_of(self, op: WorkerOperation, workers: Iterable['AnnotatedSocket']):
        for i, worker in enumerate(workers):
            worker.sock.send(op.SerializeToString())

    def receive_from_each_of(self, workers: Iterable['AnnotatedSocket'],
                             throw_errors: bool = True) -> List[WorkerConfirm]:
        confs = []
        for i, worker in enumerate(workers):
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
        self.logger("Closing pool with {} workers".format(len(self.workers)))
        close_op = WorkerOperation()
        close_op.job_id = job_id
        close_op.close = True
        for _, _, worker in self.workers:
            worker.sock.send(close_op.SerializeToString())
        freed_workers = [w for _, _, w in self.workers]
        self.workers = []
        return freed_workers


class AnnotatedSocket:
    def __init__(self, sock, info):
        self.sock = sock
        self.info = info

    def fileno(self):
        return self.sock.fileno()


class ManagerServer:
    def __init__(self, host: str, port: int, manager: Manager,
                 certfile: str = None, keyfile: str = None, logger: ServerLogger = None):
        if logger is None:
            self.logger = PrintLogger()
        else:
            self.logger = logger

        self.manager = manager

        self.tcpsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcpsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.tcpsock.bind((host, port))

        self.ssl = certfile and keyfile
        self.certfile = certfile
        self.keyfile = keyfile

        self.timeout_duration = 1

        self.running = True

    def run(self):
        self.tcpsock.listen(5)
        self.logger.starting_server()
        while self.running:
            self.logger.accepting_connections()
            clientsock, (ip, port) = self.tcpsock.accept()
            clientsock.settimeout(self.timeout_duration)
            try:
                self.logger.accepted_connection(bool(self.ssl))

                clientsock.send(b'\x01' if self.ssl else b'\x00')
                if self.ssl:
                    clientsock = ssl.wrap_socket(
                        clientsock,
                        server_side=True,
                        certfile=self.certfile,
                        keyfile=self.keyfile
                    )
                clientformatsock = formatsock.FormatSocket(clientsock)
                clientformatsock.settimeout(None)
                host_info = HostInformation.FromString(clientformatsock.recv())
                host_info.address = ip
                host_info.port = port
                t = Thread(target=handle, args=(self, clientformatsock, host_info))
                t.start()
            except IOError as e:
                self.logger.log_error("Error accepting connection: {}".format(str(e)))
                try:
                    clientsock.close()
                except IOError as e:
                    self.logger.log_error("Error while closing socket: {}".format(str(e)))

    def stop(self):
        self.running = False
        self.manager.close()


def handle(server: ManagerServer, sock: socket, host_info: HostInformation):
    if host_info.HasField('worker_info'):
        server.logger.received_worker(host_info)
        server.manager.add_worker(sock, host_info)
    elif host_info.HasField('client_info'):
        server.logger.received_client(host_info)
        server.manager.add_client(sock, host_info)
        server.manager.serve_client(sock)


if __name__ == "__main__":
    man = Manager()
    manserver = ManagerServer('0.0.0.0', 1708, man)
    manserver.run()
