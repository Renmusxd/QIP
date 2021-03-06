import numpy
from qip.util import flatten
from qip.backend import StateType, InitialState, CythonBackend
from typing import Mapping, Sequence, Tuple, Callable, AbstractSet, Any, Union, Type


class PipelineObject(object):
    def __init__(self, quantum: bool, default: InitialState = None, n: int = 0):
        self.quantum = quantum
        self.inputs = []
        self.sink = []
        self.default = default
        # PipelineObjects don't be default contribute to qubit circuit.
        self.n = n

    def run(self, feed: Mapping['PipelineObject', InitialState] = None, **kwargs):
        return run(self, feed=feed, **kwargs)

    def feed(self, state: StateType, qbitindex: Mapping['PipelineObject', Sequence[int]],
             n: int) -> Tuple[int, object]:
        """
        Operate on the state of the system.
        :param state: input state
        :param qbitindex: mapping of qbit to global index
        :param n: number of qubits
        :param backend: backend to use for matrix operations
        :return: (state, (num classic bits, int bits))
        """
        return self.feed_indices(state, [qbitindex[q] for q in self.inputs], n)

    def feed_indices(self, state: StateType, index_groups: Sequence[Sequence[int]],
                     n: int) -> Tuple[int, Any]:
        """
        Operate on the state of the system.
        :param state: input state
        :param index_groups: array of arrays of indicies used by each input in order.
        :param n: number of qubits
        :param backend: backend to use for matrix operations
        :return: (num qubits removed, some operation annotation)
        """
        # Return identity
        return 0, 0

    def select_index(self, indices: Sequence[int]) -> Sequence[int]:
        """
        May be overridden to modify index selection (see SplitQubit).
        :param indices: list of indices from all inputs nodes
        :return:
        """
        return indices

    def remap_index(self, index_map: Mapping['PipelineObject', int], n: int) -> Mapping['PipelineObject', int]:
        """
        May be override to rearrange qubits if needed
        :param index_map: map of Qubit -> Index
        :param n: n qubits in system
        :return: new index_map
        """
        return index_map

    def set_sink(self, sink: 'PipelineObject'):
        self.sink += [sink]

    def get_inputs(self) -> Sequence['PipelineObject']:
        return self.inputs

    def __hash__(self):
        return hash(repr(self))


def run(*args: PipelineObject,
        feed: Mapping[Union[PipelineObject, Tuple[PipelineObject, ...]], InitialState] = None,
        strict: bool = False,
        backend_constructor: Callable[[int, Sequence[Tuple[int, ...]], Sequence[Sequence[complex]]], StateType] = None,
        statetype: Type = numpy.complex128,
        **kwargs):
    """
    Runs pipeline using all qubits in *args. Produces an output state based on input.
    :param args: list of qubits to evaluate
    :param feed: feed of individual qbit states - qbit/qbits : state
    :param statetype: type of state data (should be complex numpy value).
    :param strict: requires all qubits to have states in feed, none implicitly defined as |0>
    :param backend_constructor
    :return: state handle, output dictionary
    """
    if feed is None:
        feed = {}
    else:
        # Copy to dictionary, convert non-tuples to tuples
        feed = {(k if type(k) == tuple else (k,)): feed[k] for k in feed}

    # Flatten keys for the ones which are qubits
    all_qubits_in_feed = list(flatten(feed.keys()))

    if backend_constructor is None:
        backend_constructor = CythonBackend.make_state

    # Frontier contains all qubits required for execution
    # Assume 0 unless in feed
    frontier, graphnodes = get_deps(*args, feed=feed)
    for qubit in frontier:
        if qubit not in all_qubits_in_feed and qubit.default is not None:
            if type(qubit.default) == int:
                if 0 < qubit.default < 2**qubit.n:
                    feed[(qubit,)] = numpy.zeros((2**qubit.n,))
                    feed[(qubit,)][qubit.default] = 1.0
                # if it's 0 then not defining it is faster
            else:
                feed[(qubit,)] = qubit.default

    frontier = list(sorted(frontier, key=lambda q: q.qid))

    if strict:
        missing_qubits = [qubit for qubit in frontier if qubit not in all_qubits_in_feed]
        if len(missing_qubits):
            raise ValueError("Missing Qubit states for: {}".format(missing_qubits))

    qbitindex = {}
    n = 0
    for qbit in frontier:
        qbitindex[qbit] = [i for i in range(n, n + qbit.n)]
        n += qbit.n

    feed_index_groups = [sum((qbitindex[q] for q in qs), []) for qs in feed]
    feed_states = [feed[qs] for qs in feed]

    # In case they used the deprecated state argument
    # Convert state into a single index group for all indices.
    if 'state' in kwargs and kwargs['state']:
        feed_index_groups = [tuple(range(n))]
        feed_states = [kwargs['state']]

    state = backend_constructor(n, feed_index_groups, feed_states, statetype=statetype)

    return feed_forward(frontier, state, graphnodes)


def get_deps(*args: PipelineObject, feed: Mapping[PipelineObject, InitialState] = None) \
        -> Tuple[Sequence[PipelineObject], AbstractSet[PipelineObject]]:
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


def make_circuit_mat(*args: PipelineObject) -> numpy.ndarray:
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
    def feed(self, qbitindex: Mapping[PipelineObject, Sequence[int]], node: PipelineObject) -> None:
        raise NotImplemented("Method not implemented.")


def run_graph(frontier: Sequence[PipelineObject], graphnodes: AbstractSet[PipelineObject], graphacc: GraphAccumulator):
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
    def __init__(self, state: StateType, n: int):
        self.state = state
        self.n = n
        self.classic_map = {}

        # TODO make state agnostic version
        # if len(self.state) != 2 ** n:
        #     raise ValueError("Size of state must be 2**n")

    def feed(self, qbitindex: Mapping[PipelineObject, Sequence[int]], node: PipelineObject):
        n_bits, bits = node.feed(self.state, qbitindex, self.n)

        if n_bits > 0 or bits is not None:
            self.classic_map[node] = bits
            self.n -= n_bits


def feed_forward(frontier: Sequence[PipelineObject], state: StateType,
                 graphnodes: AbstractSet[PipelineObject]):
    n = sum(f.n for f in frontier)
    graphacc = NodeFeeder(state, n)
    run_graph(frontier, graphnodes, graphacc)
    return graphacc.state.get_state(), graphacc.classic_map


class PrintFeeder(GraphAccumulator):
    BLACKLIST = ["SplitQubit", "Q"]

    def __init__(self, n: int, opwidth: int = 1, linespacing: int = 1, outputfn: Callable[[str], None] = print):
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


def print_circuit(*args, opwidth: int = 1, linespacing: int = 1, outputfn: Callable[[str], None] = print):
    frontier, graphnodes = get_deps(*args)
    frontier = list(sorted(frontier, key=lambda q: q.qid))
    n = sum(f.n for f in frontier)
    graphacc = PrintFeeder(n, opwidth, linespacing, outputfn)
    run_graph(frontier, graphnodes, graphacc)