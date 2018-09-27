from qip.distributed.proto import *
import time


class WorkerLogger(object):
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        if len(args) == 1:
            self.log_string(str(args[0]), **kwargs)
        elif len(args) > 1:
            self.log_string(str(args), **kwargs)

    def log_string(self, s, **kwargs):
        raise NotImplemented("Base class not implemented")

    def log_error(self, s, **kwargs):
        self.log_string("[!] " + s, **kwargs)

    def starting_server(self):
        self.log_string("Starting server")

    def accepting_connections(self):
        self.log_string("Accepting connections.")

    def accepted_connection(self):
        self.log_string("Accepted connection")

    def waiting_for_setup(self):
        self.log_string("Waiting for setup.")

    def accepted_setup(self, setup: WorkerSetup):
        self.log_string("Setup: {}".format(setup))

    def making_state(self, handle: str):
        self.log_string("Making state {}".format(handle))

    def closing_state(self, handle: str):
        self.log_string("Closing client {}".format(handle))

    def waiting_for_operation(self, handle: str):
        self.log_string("Waiting for operation for job {}".format(handle))

    def running_operation(self, handle: str, op: WorkerOperation):
        self.log_string("Running operation: {} for {}".format(op, handle))

    def done_running_operation(self, handle: str, op: WorkerOperation):
        self.log_string("Done running operation for {}".format(handle))


class PrintLogger(WorkerLogger):
    def __init__(self, print_fn=print):
        super().__init__()
        self.print = print_fn

    def log_string(self, s, **kwargs):
        timestruct = time.localtime()
        self.print("[{:02d}:{:02d}:{:02d}] {}".format(timestruct.tm_hour, timestruct.tm_min, timestruct.tm_sec, s),
                   **kwargs)
