from qip.distributed.backend import DistributedBackend


d = DistributedBackend(n=10)
state = d.make_state([(0,)], [[1.0, 0.0]])
d.kronselect_dot({(0,): [[0.0, 1.0],[1.0, 0.0]]}, state, 10)

d.close()