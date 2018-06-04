import numpy
from qip.util import gen_edit_indices
from qip.util import flatten


def run(*args, state=None, feed=None, statetype=numpy.complex128, debug=False, strict=False, **kwargs):
    """
    Runs pipeline using all qubits in *args
    :param args: list of qubits to evaluate
    :param state: full state of n qbits
    :param feed: feed of individual qbit states
    :return: state of full qubit system (2**n array)
    """
    if feed is None:
        feed = {}
    else:
        feed = feed.copy()

    # Frontier contains all qubits required for execution
    # Assume 0 unless in feed
    frontier, graphnodes = get_deps(*args, feed=feed)
    for qubit in frontier:
        if qubit not in feed and qubit.default is not None:
            if type(qubit.default) == int:
                if 0 < qubit.default < 2**qubit.n:
                    feed[qubit] = numpy.zeros((2**qubit.n,))
                    feed[qubit][qubit.default] = 1.0
                # if it's 0 then not defining it is faster
            else:
                feed[qubit] = qubit.default

    qbits = list(sorted(feed.keys(), key=lambda q: q.qid))
    frontier = list(sorted(frontier, key=lambda q: q.qid))

    if strict:
        missing_qubits = [qubit for qubit in frontier if qubit not in feed]
        if len(missing_qubits):
            raise ValueError("Missing Qubit states for: {}".format(missing_qubits))
    # else:
    #     undefined qubits default to |0>

    n = sum(q.n for q in frontier)
    if state is None:
        qbitindex = {}
        n = 0
        for qbit in sorted(frontier, key=lambda q: q.qid):
            qbitindex[qbit] = [i for i in range(n, n + qbit.n)]
            n += qbit.n

        state = numpy.zeros(2**n, dtype=statetype)

        if len(qbits) == 0:
            state[0] = 1

        # Set all the entries in state to product of matrix entries
        for index, flips in gen_edit_indices([qbitindex[qbit] for qbit in qbits], n-1):
            state[index] = 1.0
            for qindex, flip in enumerate(flips):
                qbit = qbits[qindex]
                state[index] = state[index] * feed[qbit][flip]
    elif type(state) == int:
        stateint = state
        state = numpy.zeros(2**n, dtype=statetype)
        state[stateint] = 1.0
    elif len(state) != 2**n:
        raise ValueError("State size must be 2**n")
    elif type(state) == list:
        state = numpy.array(state)

    return feed_forward(frontier, state, graphnodes, debug=debug, statetype=statetype)


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


def feed_forward(frontier, state, graphnodes, statetype=numpy.complex128, debug=False):
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

    state = state.astype(statetype)
    arena = numpy.ndarray(shape=(2**n,), dtype=statetype)
    classic_map = {}

    while len(frontier) > 0:
        frontier = list(sorted(frontier, key=lambda q: q.qid))
        node, frontier = frontier[0], frontier[1:]

        if debug:
            print(node)

        state, arena, (n_bits, bits) = node.feed(state, qbitindex, n, arena)

        # Manage measurements
        qbitindex = node.remap_index(qbitindex, n)

        if n_bits > 0 or bits is not None:
            classic_map[node] = bits
            n -= n_bits

        seen.add(node)
        # Iterate sink for special cases where "cloning" takes place (i.e. splitting qubits after operation)
        for nextnode in node.sink:
            if nextnode in graphnodes and nextnode not in seen and nextnode not in frontier:
                all_deps = True
                for prevnode in nextnode.inputs:
                    if prevnode not in seen:
                        all_deps = False
                        break
                if all_deps:
                    frontier.append(nextnode)
                    qbitindex[nextnode] = nextnode.select_index(flatten(qbitindex[j] for j in nextnode.inputs))
    return state, classic_map


def make_circuit_mat(*args):
    """
    Gives matrix for quantum part of circuit, ignores measurements.
    :param args: outputs (like run(...))
    :return: matrix.
    """
    # First do 0 state to get n
    o, _ = run(*args)  # Pretend classic measurement doesn't exist
    n = int(numpy.log2(len(o)))

    mat = numpy.zeros((2**n, 2**n), dtype=numpy.complex128)
    mat[:,0] = o
    for i in range(1,2**n):
        o, _ = run(*args, state=i)
        mat[:,i] = o
    return mat

blacklist = ["SplitQubit", "Q"]
def printCircuit(*args, opwidth=1, linespacing=1, outputfn=print):
    """
    Print out circuit which leads to qubits in args
    :param args: list of qubits (same as would be passed to run(*args)
    """
    frontier, graphnodes = get_deps(*args)
    frontier = list(sorted(frontier, key=lambda q: q.qid))

    qbitindex = {}
    seen = set()
    n = 0
    for qbit in frontier:
        qbitindex[qbit] = [i for i in range(n, n + qbit.n)]
        n += qbit.n

    qubit_line = (" "*opwidth) + "|" + (" "*opwidth)
    while len(frontier) > 0:
        frontier = list(sorted(frontier, key=lambda q: q.qid))
        node, frontier = frontier[0], frontier[1:]

        indices = flatten([qbitindex[qubit] for qubit in node.inputs])
        if len(indices) > 0:
            nodestr = repr(node)
            paren_pos = nodestr.find('(')
            if paren_pos > 0:
                nodestr = nodestr[:paren_pos]

            if nodestr not in blacklist:
                # print node at relevant positions
                default_line = [qubit_line] * n

                outputfn((" " * linespacing).join(default_line))

                for index in indices:
                    default_line[index] = "-"*(2*opwidth + 1)
                outputfn((" "*linespacing).join(default_line))

                for nodechr in nodestr:
                    default_line = [qubit_line]*n
                    for index in indices:
                        default_line[index] = "|" + (" "*(opwidth-1)) + nodechr + (" "*(opwidth-1)) + "|"
                    outputfn((" " * linespacing).join(default_line))

                max_len = max(len(str(i)) for i in range(len(indices)))
                index_strs = []
                for i in range(len(indices)):
                    index_str = str(i)
                    difflen = max_len - len(index_str)
                    if difflen > 0:
                        index_str = " "*difflen + index_str
                    index_strs.append(index_str)
                for l in range(max_len):
                    for i,index in enumerate(indices):
                        default_line[index] = "|" + (" " * (opwidth - 1)) + index_strs[i][l] + (" " * (opwidth - 1)) + "|"
                    outputfn((" " * linespacing).join(default_line))

                for index in indices:
                    default_line[index] = "-"*(2*opwidth + 1)
                outputfn((" "*linespacing).join(default_line))

        # Manage measurements
        qbitindex = node.remap_index(qbitindex, n)
        seen.add(node)

        # Iterate sink for special cases where "cloning" takes place (i.e. splitting qubits after operation)
        for nextnode in node.sink:
            if nextnode in graphnodes and nextnode not in seen and nextnode not in frontier:
                all_deps = True
                for prevnode in nextnode.inputs:
                    if prevnode not in seen:
                        all_deps = False
                        break
                if all_deps:
                    frontier.append(nextnode)
                    qbitindex[nextnode] = nextnode.select_index(flatten(qbitindex[j] for j in nextnode.inputs))
