from qip.qip import Qubit, OpConstructor
from qip.util import flatten
import collections
from typing import Any, Sequence, Callable, Tuple, Union, Optional, Mapping, Type


class QubitWrapperContext:
    TOTAL_CONTEXT_QUBITS = 0
    CONTEXT_STACK = []

    def __init__(self, *args: Union[Callable[[OpConstructor], OpConstructor], Sequence[Qubit], Sequence[int]]):
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

    def add_context(self, constructor: OpConstructor, qubits: Sequence[Qubit], indices: Optional[Sequence[int]] = None):
        if indices is not None:
            self.context.append((constructor, qubits, indices))
        else:
            self.context.append((constructor, qubits, list(range(len(qubits)))))
        self.context_size += len(qubits)

    @staticmethod
    def apply_to_qubits_in_context(op_func: OpConstructor, *qubits: Qubit,
                                   op_args: Optional[Sequence[Any]] = None,
                                   op_kwargs: Optional[Mapping[str, Any]] = None) -> Tuple[Qubit, ...]:
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
            op_func = QubitWrapperContext.make_context_constructor(op_func, op_args=op_args, op_kwargs=op_kwargs)

        # Dont change dict in case it's being reused
        op_kwargs = op_kwargs.copy() if op_kwargs else {}
        # Any other ops built by this call should not have context reapplied.
        op_kwargs['nocontext'] = True
        op_kwargs['nosplit'] = True
        qubits_list = op_func(*qubits_list, **op_kwargs).split()

        if len(QubitWrapperContext.CONTEXT_STACK) > 0:
            applied_control_qubits, qubits_list = QubitWrapperContext.split_context_and_circuit_qubits(*qubits_list)
            QubitWrapperContext.set_qubits(applied_control_qubits)

        return qubits_list

    def put_qubits_in_local_context_order(self, *qubits: Qubit) -> Tuple[Qubit, ...]:
        qubits_list = list(qubits)
        for _, context_qubits, context_indices in reversed(self.context):
            for c_qubit, c_index in zip(context_qubits, context_indices):
                qubits_list.insert(c_index, c_qubit)
        return tuple(qubits_list)

    @staticmethod
    def put_qubits_in_context_order(*qubits: Qubit) -> Tuple[Qubit, ...]:
        all_context = QubitWrapperContext.get_context()
        qubits_list = list(qubits)
        for _, context_qubits, context_indices in reversed(all_context):
            for c_qubit, c_index in zip(context_qubits, context_indices):
                qubits_list.insert(c_index, c_qubit)
        return tuple(qubits_list)

    @staticmethod
    def split_context_and_circuit_qubits(*qubits: Qubit) -> Tuple[Sequence[Qubit], Sequence[Qubit]]:
        all_context = QubitWrapperContext.get_context()
        mutable_qubits = list(qubits)
        applied_control_qubits = []
        for context in all_context:
            for c_index in context[2]:
                applied_control_qubits.append(mutable_qubits.pop(c_index))
        return applied_control_qubits, mutable_qubits

    @staticmethod
    def make_context_constructor(op_func: OpConstructor, op_args: Optional[Sequence[Any]] = None,
                                 op_kwargs: Optional[Mapping[str, Any]] = None) -> OpConstructor:
        # TODO deal with op_args/op_kwargs going to first op_func properly
        # need to feed to constructor at base, but not the ones above that.
        all_context = QubitWrapperContext.get_context()
        for context_constructor, context_qubits, context_indices in reversed(all_context):
            op_func = context_constructor(op_func)
        return op_func

    @staticmethod
    def get_context() -> Sequence[Tuple[OpConstructor, Sequence[Qubit], Optional[Sequence[int]]]]:
        """Get full set of contexts."""
        return flatten(QubitWrapperContext.CONTEXT_STACK)

    @staticmethod
    def set_qubits(qubits_list: Sequence[Qubit]):
        if len(qubits_list) != QubitWrapperContext.TOTAL_CONTEXT_QUBITS:
            raise ValueError("Size of qubits_list incorrect: {} is not required {}"
                             .format(len(qubits_list), QubitWrapperContext.TOTAL_CONTEXT_QUBITS))
        n = 0
        for context_set in QubitWrapperContext.CONTEXT_STACK:
            for i, context in enumerate(context_set):
                new_qubits = qubits_list[n:n+len(context[1])]
                context_set[i] = (context[0], new_qubits, context[2])
                n += len(new_qubits)

    def __enter__(self):
        QubitWrapperContext.CONTEXT_STACK.append(self.context)
        QubitWrapperContext.TOTAL_CONTEXT_QUBITS += self.context_size
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        QubitWrapperContext.CONTEXT_STACK.pop()
        QubitWrapperContext.TOTAL_CONTEXT_QUBITS -= self.context_size


class QubitOpWrapper(OpConstructor):
    """
    Class which wraps normal ops and allows them to split output upon call.
    """
    def __init__(self, op: Union[Type[Qubit], Callable[[Sequence[Any]], Qubit]], *args, **kwargs):
        super().__init__(op)
        self.args = args
        self.kwargs = kwargs

    def __call__(self, *inputs: Qubit,
                 nosplit: bool = False, nocontext: bool = False, **kwargs) -> Union[Qubit, Tuple[Qubit, ...]]:
        kwargs = self.kwargs.copy()
        kwargs.update(kwargs)

        if nocontext:
            q = self.op(*self.args, *inputs, nocontext=nocontext, **kwargs)
            if len(q.inputs) == 1 or nosplit:
                return q
            elif len(q.inputs) > 1:
                return q.split()
        else:
            n = QubitWrapperContext.apply_to_qubits_in_context(self, *inputs)
            if len(n) > 1:
                return n
            elif len(n) == 1:
                return n[0]
            else:
                raise ValueError("Somehow ended up with zero output qubits.")

    def wrap_op_hook(self, opconstructor: Callable[[OpConstructor], OpConstructor],
                     consumed_inputs: Optional[Sequence[int]] = None) -> Optional[OpConstructor]:
        return None

    def __repr__(self):
        if len(self.args) == 0:
            return "OpWrap({})".format(self.op)
        else:
            return "OpWrap({},+{} args)".format(self.op, len(self.args))


class QubitFuncWrapper(OpConstructor):
    def __init__(self, func: Callable[[Sequence[Any]], Union[Qubit, Tuple[Qubit, ...]]]):
        super().__init__(func)
        self.wrapper_funcs = []

    def __call__(self, *inputs: Qubit, **kwargs) -> Union[Qubit, Tuple[Qubit, ...]]:
        # Extract consumed ops in reverse order
        input_list = list(inputs)
        ops_and_qubits = []
        for opconstructor, consumed_indices in reversed(self.wrapper_funcs):
            # Remove in reverse order to preserve index values.
            consumed_qubits = reversed([input_list.pop(i) for i in reversed(consumed_indices)])
            ops_and_qubits.append((opconstructor, consumed_qubits, consumed_indices))
        ops_and_qubits = list(reversed(ops_and_qubits))

        # Qubits left in input_list are destined to go to circuit func
        # Use func to construct the circuit from args
        with QubitWrapperContext(*flatten(ops_and_qubits)) as context:
            outputs = self.op(*input_list, **kwargs)
            if isinstance(outputs, Qubit):
                outputs = (outputs,)
            in_context_qubits = context.put_qubits_in_local_context_order(*outputs)
            if len(in_context_qubits) == 1:
                return in_context_qubits[0]
            else:
                return in_context_qubits

    def wrap_op_hook(self, opconstructor: Callable[[OpConstructor], OpConstructor],
                     consumed_inputs: Optional[Sequence[int]] = None) -> OpConstructor:
        self.wrapper_funcs.append((opconstructor, consumed_inputs))
        return self

    @staticmethod
    def wrap(op_func: Callable[[Sequence[Any]], Qubit]):
        return QubitFuncWrapper(op_func)
