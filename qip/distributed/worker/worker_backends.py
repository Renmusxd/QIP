from qip.distributed.proto import WorkerSetup, WorkerOperation, WorkerConfirm, WorkerCommand
from typing import Optional, Sequence


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

    def report_probability(self, job_id: str, measured_bits: Optional[int] = None,
                           measured_prob: Optional[float] = None):
        raise NotImplemented("Not implemented")

    def report_top_probs(self, job_id: str, top_indices: Sequence[int], top_probs: Sequence[float]):
        raise NotImplemented("Not implemented")


class SocketServerBackend(ServerBackend):
    def __init__(self, serversocket):
        super().__init__()
        self.serversocket = serversocket

    def receive_command(self):
        return WorkerCommand.FromString(self.serversocket.recv())

    def receive_operation(self):
        return WorkerOperation.FromString(self.serversocket.recv())

    def report_done(self, job_id: str):
        conf = WorkerConfirm()
        conf.job_id = job_id
        self.serversocket.send(conf.SerializeToString())

    def report_probability(self, job_id: str, measured_bits: Optional[int] = None,
                           measured_prob: Optional[float] = None):
        conf = WorkerConfirm()
        conf.job_id = job_id
        if measured_bits is not None:
            conf.measure_result.measured_bits = measured_bits
        if measured_prob is not None:
            conf.measure_result.measured_prob = measured_prob
        self.serversocket.send(conf.SerializeToString())

    def report_top_probs(self, job_id: str, top_indices: Sequence[int], top_probs: Sequence[float]):
        conf = WorkerConfirm()
        conf.job_id = job_id
        conf.measure_result.top_k_indices.index.extend(top_indices)
        conf.measure_result.top_k_probs.extend(top_probs)
        self.serversocket.send(conf.SerializeToString())
