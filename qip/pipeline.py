import numpy
from qip.util import gen_edit_indices
from qip.util import flatten
from qip.backend import CythonBackend


def run(*args, state=None, feed=None, statetype=numpy.complex128, strict=False, backend_constructor=CythonBackend, **kwargs):
    """
    Runs pipeline using all qubits in *args. Produces an output state based on input.
    :param args: list of qubits to evaluate
    :param state: full state of n qbits
    :param feed: feed of individual qbit states
    :param statetype: type of state data (should be complex numpy value).
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

    qbitindex = {}
    n = 0
    for qbit in sorted(frontier, key=lambda q: q.qid):
        qbitindex[qbit] = [i for i in range(n, n + qbit.n)]
        n += qbit.n

    backend = backend_constructor(n)

    feed_index_groups = [qbitindex[qbit] for qbit in qbits]
    feed_states = [feed[qbits[qindex]] for qindex in range(len(qbits))]

    state, arena = backend.make_state(feed_index_groups, feed_states,
                                      state=state, statetype=statetype)

    return feed_forward(frontier, state, arena, graphnodes, backend)


def get_deps(*args, feed=None):
    """
    Gets all graph nodes required for evaluation of graph.
    :param args: desired output nodes.
    :param feed: nodes with preassigned values.
    :return: list of top level nodes and list of all nodes.
    """
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


class GraphAccumulator:
    def feed(self, qbitindex, node):
        raise NotImplemented("Method not implemented.")


def run_graph(frontier, graphnodes, graphacc):
    """
    Apply the feed function from graphacc to each node in the graph in the order necessary to run the circuit.
    :param frontier: top level nodes of graph
    :param graphnodes: all nodes in graph
    :param graphacc: graph accumulator class with "feed(qbitindex, node) -> void" function.
    """
    # Condition is that if a node is in the frontier, either all its inputs are
    # in the feed dict, or it itself is.
    qbitindex = {}
    seen = set()
    n = 0
    for qbit in sorted(frontier, key=lambda q: q.qid):
        qbitindex[qbit] = [i for i in range(n, n + qbit.n)]
        n += qbit.n

    while len(frontier) > 0:
        frontier = list(sorted(frontier, key=lambda q: q.qid))
        node, frontier = frontier[0], frontier[1:]

        graphacc.feed(qbitindex, node)

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


class NodeFeeder(GraphAccumulator):
    def __init__(self, state, arena, n, backend):
        self.state = state
        self.arena = arena
        self.n = n
        self.classic_map = {}
        self.backend = backend

        if len(self.state) != 2 ** n:
            raise ValueError("Size of state must be 2**n")

    def feed(self, qbitindex, node):
        self.state, self.arena, (n_bits, bits) = node.feed(self.state, qbitindex, self.n, self.arena, self.backend)

        if n_bits > 0 or bits is not None:
            self.classic_map[node] = bits
            self.n -= n_bits


def feed_forward(frontier, state, arena, graphnodes, backend):
    n = sum(f.n for f in frontier)
    graphacc = NodeFeeder(state, arena, n, backend)
    run_graph(frontier, graphnodes, graphacc)
    return graphacc.state, graphacc.classic_map


class PrintFeeder(GraphAccumulator):
    BLACKLIST = ["SplitQubit", "Q"]

    def __init__(self, n, opwidth=1, linespacing=1, outputfn=print):
        self.opwidth = opwidth
        self.linespacing = linespacing
        self.outputfn = outputfn
        self.n = n
        self.qubit_line = (" "*self.opwidth) + "|" + (" "*self.opwidth)

    def feed(self, qbitindex, node):
        indices = flatten([qbitindex[qubit] for qubit in node.inputs])
        if len(indices) > 0:
            nodestr = repr(node)
            paren_pos = nodestr.find('(')
            if paren_pos > 0:
                nodestr = nodestr[:paren_pos]

            if nodestr not in PrintFeeder.BLACKLIST:
                # print node at relevant positions
                default_line = [self.qubit_line] * self.n

                self.outputfn((" " * self.linespacing).join(default_line))

                for index in indices:
                    default_line[index] = "-" * (2 * self.opwidth + 1)
                self.outputfn((" " * self.linespacing).join(default_line))

                for nodechr in nodestr:
                    default_line = [self.qubit_line] * self.n
                    for index in indices:
                        default_line[index] = "|" + (" " * (self.opwidth - 1)) +\
                                              nodechr + (" " * (self.opwidth - 1)) + "|"
                    self.outputfn((" " * self.linespacing).join(default_line))

                max_len = max(len(str(i)) for i in range(len(indices)))
                index_strs = []
                for i in range(len(indices)):
                    index_str = str(i)
                    difflen = max_len - len(index_str)
                    if difflen > 0:
                        index_str = " " * difflen + index_str
                    index_strs.append(index_str)
                for l in range(max_len):
                    for i, index in enumerate(indices):
                        default_line[index] = "|" + (" " * (self.opwidth - 1)) + index_strs[i][l] + (
                                    " " * (self.opwidth - 1)) + "|"
                    self.outputfn((" " * self.linespacing).join(default_line))

                for index in indices:
                    default_line[index] = "-" * (2 * self.opwidth + 1)
                self.outputfn((" " * self.linespacing).join(default_line))


def printCircuit(*args, opwidth=1, linespacing=1, outputfn=print):
    frontier, graphnodes = get_deps(*args)
    frontier = list(sorted(frontier, key=lambda q: q.qid))
    n = sum(f.n for f in frontier)
    graphacc = PrintFeeder(n, opwidth, linespacing, outputfn)
    run_graph(frontier, graphnodes, graphacc)