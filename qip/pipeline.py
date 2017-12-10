import numpy
from qip.util import gen_edit_indices
from qip.util import flatten


def run(*args, state=None, feed=None, **kwargs):
    """
    Runs pipeline using all qubits in *args
    :param args: list of qubits to evaluate
    :param state: full state of n qbits
    :param feed: feed of individual qbit states
    :return: state of full qubit system (2**n array)
    """
    if feed is None:
        feed = {}
    qbits = list(sorted(feed.keys(), key=lambda q: q.qid))

    # Frontier contains all qubits required for execution
    # Assume 0 unless in feed
    frontier, graphnodes = get_deps(*args, feed=feed)
    frontier = list(sorted(frontier, key=lambda q: q.qid))

    n = sum(q.n for q in frontier)
    if state is None:
        qbitindex = {}
        n = 0
        for qbit in sorted(frontier, key=lambda q: q.qid):
            qbitindex[qbit] = [i for i in range(n, n + qbit.n)]
            n += qbit.n

        state = numpy.zeros(2**n, dtype=numpy.complex128)

        # Set all the entries in state to product of matrix entries
        state[0] = 1.0
        for index, flips in gen_edit_indices([qbitindex[qbit] for qbit in qbits]):
            state[index] = 1.0
            for qindex, flip in enumerate(flips):
                qbit = qbits[qindex]
                state[index] = state[index] * feed[qbit][flip]
    elif len(state) != 2**n:
        raise ValueError("State size must be 2**n")

    debug = False
    if 'debug' in kwargs:
        debug = kwargs['debug']

    return feed_forward(frontier, state, graphnodes, debug=debug)


def get_deps(*args, feed=None):
    if feed is None:
        feed = {}

    seen = set()
    deps = set()
    frontier = set(args)
    # Populate frontier
    while len(frontier) > 0:
        node = frontier.pop()
        if node in feed or len(node.inputs) == 0:
            deps.add(node)
        else:
            frontier.update(node.inputs)
        seen.add(node)
    return list(deps), seen


def feed_forward(frontier, state, graphnodes, debug=False):
    # Condition is that if a node is in the frontier, either all its inputs are
    # in the feed dict, or it itself is.
    qbitindex = {}
    seen = set()
    n = 0
    for qbit in sorted(frontier, key=lambda q: q.qid):
        qbitindex[qbit] = [i for i in range(n, n+qbit.n)]
        n += qbit.n

    if len(state) != 2**n:
        raise ValueError("Size of state must be 2**n")

    state = state.astype(numpy.complex128)
    arena = numpy.ndarray(shape=(2**n,), dtype=numpy.complex128)
    classic_map = {}

    while len(frontier) > 0:
        node = frontier.pop()

        if debug:
            print(node)

        state, arena, (n_bits, bits) = node.feed(state, qbitindex, n, arena)

        # Manage measurements
        qbitindex = node.remap_index(qbitindex, n)

        if n_bits > 0:
            classic_map[node] = bits
            n -= n_bits

        seen.add(node)
        # Iterate sink for special cases where "cloning" takes place (i.e. splitting qubits after operation)
        for nextnode in node.sink:
            if nextnode in graphnodes and nextnode not in seen:
                all_deps = True
                for prevnode in nextnode.inputs:
                    if prevnode not in seen:
                        all_deps = False
                        break
                if all_deps:
                    frontier.append(nextnode)
                    qbitindex[nextnode] = nextnode.select_index(flatten(qbitindex[j] for j in nextnode.inputs))
    return state, classic_map
