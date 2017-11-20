import numpy
from qip.util import uint_to_bitarray


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
        state = numpy.zeros(2**len(frontier))
        state[0] = 1.0

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
    qbitindex = {frontier[i]: [i] for i in range(len(frontier))}
    n = len(frontier)
    while len(frontier) > 0:
        node = frontier.pop()
        state = node.feed(state,qbitindex,n)
        for nextnode in node.sink:
            if nextnode in graphnodes:
                frontier.append(nextnode)
                if nextnode not in qbitindex:
                    qbitindex[nextnode] = []
                qbitindex[nextnode] += qbitindex[node]
    return state
