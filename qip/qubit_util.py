from qip.qip import Qubit, OpConstructor
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
        super().__init__()
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


class QubitOpWrapper(OpConstructor):
    """
    Class which wraps normal ops and allows them to split output upon call.
    """
    def __init__(self, op, *args, **kwargs):
        super().__init__()
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


class QubitFuncWrapper(OpConstructor):
    def __init__(self, func):
        super().__init__()
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
