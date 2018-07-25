from __future__ import print_function

from qip.distributed.messages import WorkerSetup
from qip.distributed import formatsock
from threading import Thread, Lock, Condition
from threading import Lock
import time
import socket
import ssl
import select
import json


class Manager:
    INPUT_TIMEOUT = 1

    def __init__(self, logger=print):
        self.logger = logger

        self.clients = {}
        self.workers = {}
        self.client_lock = Lock()
        self.worker_lock = Lock()
        self.client_con = Condition(self.client_lock)
        self.worker_con = Condition(self.worker_lock)

    def serve(self):
        while True:
            # Workers shouldn't be talking if we don't have a client, or at least we don't care what they're saying.
            with self.client_lock:
                clients = list(self.clients.values())
                while len(clients) == 0:
                    try:
                        self.client_con.wait()
                        clients = list(self.clients.values())
                    except:
                        pass

            # Now let's get workers if we have any.
            with self.worker_lock:
                workers = list(self.workers.values())

            all_connections = clients + workers
            if len(all_connections) > 0:
                rs, _, _ = select.select(all_connections, [], [], Manager.INPUT_TIMEOUT)

                # For each connection which is ready to read.
                for connection in rs:
                    host_socket = connection.socket
                    host_info = connection.info
                    # TODO

            else:
                time.sleep(Manager.INPUT_TIMEOUT)

    def add_worker(self, workersock, info):
        with self.worker_lock:
            self.workers[info] = AnnotedSocket(workersock, info)
            self.worker_con.notify_all()

    def add_client(self, clientsock, info):
        with self.client_lock:
            self.clients[info] = AnnotedSocket(clientsock,info)
            self.client_con.notify_all()


class AnnotedSocket:
    def __init__(self, sock, info):
        self.sock = sock
        self.info = info

    def fileno(self):
        return self.sock.fileno()


class ManagerServer(Thread):
    def __init__(self, host, port, manager, certfile=None, keyfile=None, logger=print):
        Thread.__init__(self)
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
            clientsock, (ip, port) = self.tcpsock.accept()
            clientsock.settimeout(self.timeout_duration)
            try:
                self.logger("[*] Accepting connection")
                self.logger("\tSending SSL: "+('ON' if self.ssl else 'OFF'))
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
                msgbytes = clientformatsock.recv()
                host_info = json.loads(msgbytes.decode('UTF-8'))
                host_info['addr'] = ip

                if host_info['type'] == 'worker':
                    self.logger("[+] Received worker: {}".format(host_info))
                    self.manager.add_worker(clientsock, host_info)
                elif host_info['type'] == 'client':
                    self.logger("[+] Received connection: {}".format(host_info))
                    self.manager.add_connection(clientformatsock, host_info)
            except IOError as e:
                self.logger("[!] Error accepting connection: {}".format(str(e)))
                try:
                    clientsock.close()
                except:
                    pass