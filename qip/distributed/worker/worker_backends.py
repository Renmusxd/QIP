from qip.distributed.proto import WorkerSetup, WorkerOperation, WorkerConfirm
from typing import Optional


class ServerBackend:
    def __init__(self):
        pass

    def receive_setup(self) -> WorkerSetup:
        """Wait for a setup command."""
        raise NotImplemented("Not implemented")

    def receive_operation(self) -> WorkerOperation:
        """Wait for an operation."""
        raise NotImplemented("Not implemented")

    def report_done(self, job_id: str) -> None:
        """Tell the server that the task job_id is done"""
        raise NotImplemented("Not implemented")

    def report_probability(self, job_id: str, measured_prob: Optional[float] = None,
                           measured_bits: Optional[int] = None):
        raise NotImplemented("Not implemented")


class WorkerBackend:
    def __init__(self):
        pass

    def send_state(self, state):
        """Send local state shard to worker"""
        raise NotImplemented("Not implemented")

    def receive_state(self, buffer):
        """Receive new state and save to buffer"""
        raise NotImplemented("Not implemented")


# Socket implementations


class SocketServerBackend(ServerBackend):
    def __init__(self, serversocket):
        super().__init__()
        self.serversocket = serversocket

    def receive_setup(self):
        return WorkerSetup.FromString(self.serversocket.recv())

    def receive_operation(self):
        return WorkerOperation.FromString(self.serversocket.recv())

    def report_done(self, job_id: str):
        conf = WorkerConfirm()
        conf.job_id = job_id
        self.serversocket.send(conf.SerializeToString())

    def report_probability(self, job_id: str, measured_prob: Optional[float] = None,
                           measured_bits: Optional[int] = None):
        conf = WorkerConfirm()
        conf.job_id = job_id
        conf.measure_result.measured_bits = measured_bits
        conf.measure_result.measured_prob = measured_prob
        self.serversocket.send(conf.SerializeToString())


class SocketWorkerBackend(WorkerBackend):
    def __init__(self, workersocket, startindex, endindex):
        super().__init__()
        self.workersocket = workersocket
        self.startindex = startindex
        self.endindex = endindex

    # TODO Implement efficient send and receive state somehow
    def send_state(self, state):
        pass

    def receive_state(self, buffer):
        pass