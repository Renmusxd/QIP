from __future__ import print_function

import numpy
import collections


class QubitWrapperContext:
    TOTAL_CONTEXT_QUBITS = 0
    CONTEXT_STACK = []

    def __init__(self, *args):
        """
        Makes a new qubit wrapper context. Within it all calls to QubitOpWrapper made constructors will automatically
        apply op wrappers added via this context.
        :param args: args is a series of triples/doubles qubits for control, and indices those qubits
        should be placed at. If indices not included then defaults to first n where n is number of qubits.
        """
        self.context = []
        self.context_size = 0
        buff = []
        for arg in args:
            if isinstance(arg, collections.Iterable):
                buff.append(list(arg))
            else:
                if len(buff) > 0:
                    self.add_context(*buff)
                    buff = []
                buff.append(arg)
        if len(buff) > 0:
            self.add_context(*buff)

    def add_context(self, constructor, qubits, indices=None):
        if indices is not None:
            self.context.append((constructor, qubits, indices))
        else:
            self.context.append((constructor, qubits, list(range(len(qubits)))))
        self.context_size += len(qubits)

    @staticmethod
    def apply_to_qubits_in_context(op_func, *qubits, op_args=None, op_kwargs=None):
        """
        operates as a call like op_func(qubits) but applies the context constructors to op_func and arranges the
        qubits and context qubits correctly.
        :param op_func: Op contructor callable. Returns a single Qubit object or subclass.
        :param qubits: list of qubits to pass to op_func
        :param op_args: list of args to preceed qubits in call to op_func
        :param op_kwargs: keyword arguments to pass to op_func
        """
        qubits_list = list(qubits)
        if len(QubitWrapperContext.CONTEXT_STACK) > 0:
            qubits_list = QubitWrapperContext.put_qubits_in_context_order(*qubits_list)
            op_func = QubitWrapperContext.make_context_constructor(op_func)

        # Dont change dict in case it's being reused
        op_kwargs = op_kwargs.copy()
        # Any other ops built by this call should not have context reapplied.
        op_kwargs['nocontext'] = True
        op_kwargs['nosplit'] = True
        qubits_list = op_func(*op_args, *qubits_list, **op_kwargs).split()

        if len(QubitWrapperContext.CONTEXT_STACK) > 0:
            applied_control_qubits, qubits_list = QubitWrapperContext.split_context_and_circuit_qubits(*qubits_list)
            QubitWrapperContext.set_qubits(applied_control_qubits)

        return qubits_list

    def put_qubits_in_local_context_order(self, *qubits):
        qubits_list = list(qubits)
        for _, context_qubits, context_indices in reversed(self.context):
            for c_qubit, c_index in zip(context_qubits, context_indices):
                qubits_list.insert(c_index, c_qubit)
        return tuple(qubits_list)

    @staticmethod
    def put_qubits_in_context_order(*qubits):
        all_context = QubitWrapperContext.get_context()
        qubits_list = list(qubits)
        for _, context_qubits, context_indices in reversed(all_context):
            for c_qubit, c_index in zip(context_qubits, context_indices):
                qubits_list.insert(c_index, c_qubit)
        return tuple(qubits_list)

    @staticmethod
    def split_context_and_circuit_qubits(*qubits):
        all_context = QubitWrapperContext.get_context()
        mutable_qubits = list(qubits)
        applied_control_qubits = []
        for context in all_context:
            for c_index in context[2]:
                applied_control_qubits.append(mutable_qubits.pop(c_index))
        return applied_control_qubits, mutable_qubits

    @staticmethod
    def make_context_constructor(op_func):
        all_context = QubitWrapperContext.get_context()
        for context_constructor, context_qubits, context_indices in reversed(all_context):
            op_func = context_constructor(op_func)
        return op_func

    @staticmethod
    def get_context():
        """Get full set of contexts."""
        return flatten(QubitWrapperContext.CONTEXT_STACK)

    @staticmethod
    def set_qubits(qubits_list):
        if len(qubits_list) != QubitWrapperContext.TOTAL_CONTEXT_QUBITS:
            raise ValueError("Size of qubits_list incorrect: {} is not required {}"
                             .format(len(qubits_list), QubitWrapperContext.TOTAL_CONTEXT_QUBITS))
        n = 0
        for context_set in QubitWrapperContext.CONTEXT_STACK:
            for i,context in enumerate(context_set):
                new_qubits = qubits_list[n:n+len(context[1])]
                context_set[i] = (context[0], new_qubits, context[2])
                n += len(new_qubits)

    def __enter__(self):
        QubitWrapperContext.CONTEXT_STACK.append(self.context)
        QubitWrapperContext.TOTAL_CONTEXT_QUBITS += self.context_size

    def __exit__(self, exc_type, exc_val, exc_tb):
        QubitWrapperContext.CONTEXT_STACK.pop()
        QubitWrapperContext.TOTAL_CONTEXT_QUBITS -= self.context_size


class QubitOpWrapper:
    """
    Class which wraps normal ops and allows them to split output upon call.
    """
    def __init__(self, op, *args, **kwargs):
        self.op = op
        self.args = args
        self.kwargs = kwargs

    def __call__(self, *inputs, nosplit=False, nocontext=False, **kwargs):
        kwargs = self.kwargs.copy()
        kwargs.update(kwargs)

        if nocontext:
            n = self.op(*self.args, *inputs, nocontext=nocontext, **kwargs)
            if len(n.inputs) == 1 or nosplit:
                return n
            elif len(n.inputs) > 1:
                return n.split()
        else:
            n = QubitWrapperContext.apply_to_qubits_in_context(self.op, *inputs, op_args=self.args, op_kwargs=kwargs)
            if len(n) > 1:
                return n
            elif len(n) == 1:
                return n[0]
            else:
                raise ValueError("Somehow ended up with zero output qubits.")

    def wrap_op_hook(self, opconstructor, consumed_inputs=None):
        return None


class QubitFuncWrapper:
    def __init__(self, func):
        self.func = func
        self.wrapper_funcs = []

    def __call__(self, *inputs, **kwargs):
        # Extract consumed ops in reverse order
        input_list = list(inputs)
        ops_and_qubits = []
        for opconstructor, consumed_indices in reversed(self.wrapper_funcs):
            # Remove in reverse order to preserve index values.
            consumed_qubits = reversed([input_list.pop(i) for i in reversed(consumed_indices)])
            ops_and_qubits.append((opconstructor, consumed_qubits, consumed_indices))
        ops_and_qubits = list(reversed(ops_and_qubits))

        print(ops_and_qubits)

        # Qubits left in input_list are destined to go to circuit func
        # Use func to construct the circuit from args
        with QubitWrapperContext(*flatten(ops_and_qubits)) as context:
            outputs = self.func(*input_list, **kwargs)
            return context.put_qubits_in_local_context_order(outputs)

    def wrap_op_hook(self, opconstructor, consumed_inputs=None):
        self.wrapper_funcs.append((opconstructor,consumed_inputs))
        return self

    @staticmethod
    def wrap(op_func):
        return QubitFuncWrapper(op_func)


def kronselect_dot(mats, vec, n, outputarray, input_offset=0, output_offset=0, dot_impl=None):
    """
    Efficiently performs the operation: OuterProduct( m1, m2, ..., mn ) dot vec
    for the case where most mj are identity.
    :param mats: { (indices,... #m) : mat(2**m x 2**m) }
    :param vec: vector v of size 2**n
    :param n: total number of matrices (including identities)
    :param outputarray: array in which to store output
    :param dot_impl: implementation of cdot function to use, see dot_loop below for example.
    :param input_offset: offset at which vec starts relative to possible larger context
    :param output_offset: offset at which outputarray starts relative to possible larger context
    """
    if len(vec) + input_offset > 2**n:
        raise ValueError("Input vector size plus offset may be no larger than the total number of qubit states (2^n)")

    if len(outputarray) + output_offset > 2**n:
        raise ValueError("Output vector size plus offset may be no larger than the total number of qubit states (2^n)")

    newmats = {}
    for indices in mats:
        # Will be set by one of the paths
        nindices = 0
        m = None

        if type(indices) != tuple and type(indices) != int:
            raise Exception("Type of indices must be tuple: {}".format(indices))
        elif type(indices) == tuple:
            m = mats[indices]
            if type(m) == list:
                m = numpy.array(m)
            newmats[indices] = m
            nindices = len(indices)
        elif type(indices) == int:
            m = mats[indices]
            if type(m) == list:
                m = numpy.array(m)
            newmats[(indices,)] = m
            nindices = 1

        if 2**nindices != m.shape[0] or 2**nindices != m.shape[1]:
            raise Exception("Shape of square submatrix must equal 2**(number of indices): "
                            "{}: {}".format(indices, m))

    if dot_impl is not None:
        iter_indices = newmats.keys()
        cindices = numpy.array([numpy.array(x, dtype=numpy.int32) for x in iter_indices])
        cmats = numpy.array([newmats[x] if type(newmats[x]) != numpy.ndarray else newmats[x].astype(numpy.complex128)
                             for x in iter_indices])
        dot_impl(cindices, cmats, vec, n, outputarray, input_offset=input_offset, output_offset=output_offset)
    else:
        dot_loop(newmats, vec, n, outputarray, input_offset=input_offset, output_offset=output_offset)


def dot_loop(mats, vec, n, output, input_offset=0, output_offset=0):
    allindices = list(mats.keys())
    flatindices = list(sorted(set(index for indices in allindices for index in indices)))

    for outputrow in range(len(output)):
        row = outputrow + output_offset
        s = 0
        for rowcol, mijs in gen_valid_col_and_matcol(row, flatindices, n):
            r, c = rowcol
            input_col = c - input_offset
            if input_col < 0 or input_col >= len(vec):
                continue

            p = 1.0
            # Multiply required entries in each non-indentity matrix
            for indices in allindices:
                mat = mats[indices]
                submati = bitarray_to_uint([mijs[index][0] for index in indices])
                submatj = bitarray_to_uint([mijs[index][1] for index in indices])
                p *= mat[submati, submatj]

            s += p*vec[input_col]
        output[outputrow] = s
    return output


def gen_valid_col_and_matcol(row, matindices, n):
    rowbits = uint_to_bitarray(row, n)
    colbits = rowbits[:]

    matrow = tuple(rowbits[indx] for indx in matindices)
    for i in range(2**len(matindices)):
        matcol = uint_to_bitarray(i, len(matindices))
        for j, indx in enumerate(matindices):
            colbits[indx] = matcol[j]
        yield (row, bitarray_to_uint(colbits)), {matindices[j]: item for j, item in enumerate(zip(matrow, matcol))}


def gen_edit_indices(index_groups, maxindex):
    if len(index_groups) > 0:
        allindices = flatten(index_groups)

        bits = [0] * (maxindex+1)
        for i in range(2**len(allindices)):
            flips = uint_to_bitarray(i, len(allindices))
            for j, bit in enumerate(flips):
                bits[allindices[j]] = flips[j]

            qbit_state_indices = [0] * len(index_groups)
            indx = 0
            for j, index_group in enumerate(index_groups):
                subflips = flips[indx:indx+len(index_group)]
                qbit_state_indices[j] = bitarray_to_uint(subflips)
                indx += len(index_group)
            yield bitarray_to_uint(bits), qbit_state_indices


def expand_kron_matrix(mats, n):
    m = numpy.zeros((2**n, 2**n), dtype=numpy.complex128)
    mats = {i: numpy.array(mats[i], dtype=numpy.complex128) for i in mats}
    for i in range(2**n):
        v = numpy.zeros((2**n,), dtype=numpy.complex128)
        v[i] = 1.0
        kronselect_dot(mats, v, n, m[i, :])
    return m


def uint_to_bitarray(num, n):
    bits = []
    for i in range(n):
        bits.append(num % 2)
        num = num >> 1
    return bits[::-1]


def bitarray_to_uint(bits):
    s = 0
    for i in range(len(bits)):
        s += 2**i if bits[len(bits)-i-1] else 0
    return s


def flatten(lst):
    listgen = [item if isinstance(item, collections.Iterable) else (item,) for item in lst]
    return [item for sublist in listgen for item in sublist]


def qubit_index_notation(i, *qns, n=None):
    if n is None:
        n = sum(qns)
    index_array = []
    bit_array = uint_to_bitarray(i, n)
    start_indx = 0
    for qubit_size in qns:
        num = bitarray_to_uint(bit_array[start_indx:start_indx + qubit_size])
        index_array.append(num)
        start_indx += qubit_size
    return index_array


def gen_qubit_prints(state, *qs):
    qubit_sizes = []
    for q in qs:
        if type(q) == int:
            qubit_sizes.append(q)
        else:
            qubit_sizes.append(q.n)
    n = sum(qubit_sizes)
    for i in range(len(state)):
        if state[i] == 0:
            continue
        s = "|"
        s += ",".join(map(str,qubit_index_notation(i,*qubit_sizes,n=n)))
        s += "> = {}".format(state[i])
        yield s

