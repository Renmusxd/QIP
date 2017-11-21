import numpy
from qip.util import uint_to_bitarray
from qip.util import gen_edit_indices
from qip.util import flatten


def run(*args, **kwargs):
    """

    :param args:
    :param kwargs:
    :return:
    """
    state = kwargs['state'] if 'state' in kwargs else None
    feed = kwargs['feed'] if 'feed' in kwargs else {}

    # Frontier contains all qubits required for execution
    # Assume 0 unless in feed
    frontier, graphnodes = get_deps(*args, feed=feed)

    if state is None:
        qbitindex = {}
        n = 0
        for qbit in sorted(frontier, key=lambda q: q.qid):
            qbitindex[qbit] = [i for i in range(n, n + qbit.n)]
            n += qbit.n

        state = numpy.zeros(2**len(frontier))

        # Special case for all the unmentioned qbits
        state[0] = 1.0

        for index, flips in gen_edit_indices(flatten(qbitindex[qbit] for qbit in feed)):
            state[index] = 1.0
            for qindex, flip in enumerate(flips):
                qbit = frontier[qindex]
                state[index] = state[index] * feed[qbit][flip]

    return feed_forward(frontier, state, graphnodes)


def get_deps(*args, feed=None):
    if feed is None:
        feed = {}

    seen = set(feed.keys())
    deps = set()
    frontier = set(args)
    # Populate frontier
    while len(frontier) > 0:
        node = frontier.pop()
        if node in seen or len(node.inputs) == 0:
            deps.add(node)
        else:
            frontier.update(node.inputs)
        seen.add(node)
    return list(deps), seen


def feed_forward(frontier, state, graphnodes):
    # Condition is that if a node is in the frontier, either all its inputs are
    # in the feed dict, or it itself is.
    qbitindex = {}
    seen = set()
    n = 0
    for qbit in sorted(frontier, key=lambda q: q.qid):
        qbitindex[qbit] = [i for i in range(n, n+qbit.n)]
        n += qbit.n
    while len(frontier) > 0:
        node = frontier.pop()
        state = node.feed(state,qbitindex,n)
        seen.add(node)
        for nextnode in node.sink:
            if nextnode in graphnodes and nextnode not in seen:
                all_deps = True
                for prevnode in nextnode.inputs:
                    if prevnode not in seen:
                        all_deps = False
                        break
                if all_deps:
                    frontier.append(nextnode)
                    if nextnode not in qbitindex:
                        qbitindex[nextnode] = []
                    qbitindex[nextnode] += qbitindex[node]
    return state
