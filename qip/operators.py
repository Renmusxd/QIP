from qip.qip import Qubit, OpConstructor
from qip.qubit_util import QubitOpWrapper
from qip.util import flatten
from qip.backend import Backend, MatrixType, StateType, IndexType
import numpy
from typing import Sequence, Tuple, Mapping, Union, Any, Callable, cast


class MatrixOp(Qubit):
    def __init__(self, *inputs: Qubit, **kwargs):
        super().__init__(*inputs, **kwargs)
        self.ms = None

    def feed_indices(self, inputvals: StateType, index_groups: Sequence[Sequence[int]], n: int, arena: StateType,
                     backend: Backend) -> Tuple[StateType, StateType, Tuple[int, int]]:
        if self.ms is None:
            self.ms = self.makemats(index_groups)
        backend.kronselect_dot(self.ms, inputvals, n, arena)
        return arena, inputvals,  (0, 0)

    def makemats(self, index_groups: Sequence[Sequence[int]]) -> Mapping[IndexType, MatrixType]:
        """
        Make matrix to be used by the op
        :param index_groups: list of list of indices
        :return: dict from index groups to matrix
        """
        raise NotImplemented("This method should never be called.")


class NotOp(MatrixOp):
    def __init__(self, *inputs: Qubit, **kwargs):
        super().__init__(*inputs, **kwargs)

    def makemats(self, index_groups: Sequence[Sequence[int]]) -> Mapping[IndexType, MatrixType]:
        return {i: numpy.flip(numpy.eye(2), 0)
                for i in flatten(index_groups)}

    def __repr__(self) -> str:
        return "Not({})".format(self.qid)


Not = QubitOpWrapper(NotOp)


class XOp(NotOp):
    def __init__(self, *inputs: Qubit, **kwargs):
        super().__init__(*inputs, **kwargs)

    def __repr__(self):
        return "X({})".format(self.qid)


X = QubitOpWrapper(XOp)


class YOp(MatrixOp):
    def __init__(self, *inputs: Qubit, **kwargs):
        super().__init__(*inputs, **kwargs)

    def makemats(self, index_groups: Sequence[Sequence[int]]) -> Mapping[IndexType, MatrixType]:
        return {i: numpy.array([[0.0, -1.0j],[1.0j, 0.0]])
                for i in flatten(index_groups)}

    def __repr__(self) -> str:
        return "Y({})".format(self.qid)


Y = QubitOpWrapper(YOp)


class ZOp(MatrixOp):
    def __init__(self, *inputs: Qubit, **kwargs):
        super().__init__(*inputs, **kwargs)

    def makemats(self, index_groups: Sequence[Sequence[int]]) -> Mapping[IndexType, MatrixType]:
        return {i: numpy.array([[1.0, 0.0],[0.0j, -1.0]])
                for i in flatten(index_groups)}

    def __repr__(self) -> str:
        return "Z({})".format(self.qid)


Z = QubitOpWrapper(ZOp)


class HOp(MatrixOp):
    def __init__(self, *inputs: Qubit, **kwargs):
        super().__init__(*inputs, **kwargs)

    def makemats(self, index_groups: Sequence[Sequence[int]]) -> Mapping[IndexType, MatrixType]:
        return {i: (1/numpy.sqrt(2))*numpy.array([[1, 1], [1, -1]])
                for i in flatten(index_groups)}

    def __repr__(self) -> str:
        return "H({})".format(self.qid)


H = QubitOpWrapper(HOp)


class ROp(MatrixOp):
    def __init__(self, phi: float, *inputs: Qubit, **kwargs):
        super().__init__(*inputs, **kwargs)
        self.exponented = numpy.exp(1.0j * phi)

    def makemats(self, index_groups: Sequence[Sequence[int]]) -> Mapping[IndexType, MatrixType]:
        return {i: numpy.array([[1, 0], [0, self.exponented]])
                for i in flatten(index_groups)}

    def __repr__(self) -> str:
        return "R({})".format(",".join(map(repr,self.inputs)))


R = QubitOpWrapper(ROp)


class RmOp(ROp):
    def __init__(self, m: int, *inputs: Qubit, negate: bool = False, **kwargs):
        super().__init__((-2 if negate else 2) * numpy.pi / pow(2.0, m), *inputs, **kwargs)
        self.m = m

    def __repr__(self) -> str:
        return "R[{}]({})".format(self.m, ",".join(map(repr,self.inputs)))


Rm = QubitOpWrapper(RmOp)


class SwapOp(MatrixOp):
    def __init__(self, *inputs: Qubit, **kwargs):
        super().__init__(*inputs, **kwargs)
        if len(self.inputs) != 2:
            raise Exception("Swap can only take two inputs")
        if self.inputs[0].n != self.inputs[1].n:
            raise Exception("Inputs must be of equal size {}/{}".format(self.inputs[0], self.inputs[1]))

    def makemats(self, index_groups: Sequence[Sequence[int]]) -> Mapping[IndexType, MatrixType]:
        swapn = self.inputs[0].n
        a_indices = index_groups[0]
        b_indices = index_groups[1]
        return {tuple(flatten([a_indices, b_indices])): SwapMat(swapn)}

    def __repr__(self) -> str:
        return "Swap({})".format(self.qid)


Swap = QubitOpWrapper(SwapOp)


class SwapMat(object):
    """Contains special logic in the qip.ext.kronprod code"""
    _kron_struct = 3

    def __init__(self, n: int):
        """
        Constructs a 2^(2n) x 2^(2n) matrix to swap positions of blocks of n entries.
        :param n: size of swap
        """
        self.n = n
        self.shape = (2**(2*n), 2**(2*n))

    def __getitem__(self, item: Union[int, Tuple[int, int]]) -> complex:
        if type(item) == tuple and len(item) == 2:
            low_a, high_a = item[0] & ~(-1 << self.n), item[0] >> self.n
            low_b, high_b = item[1] & ~(-1 << self.n), item[1] >> self.n
            return 1.0 if low_a == high_b and low_b == high_a else 0.0

        else:
            raise ValueError("SwapMat can only be indexed with M[i,j] not M[{}]".format(item))

    def __repr__(self):
        return "SwapMat({})".format(self.n)


def C(op: OpConstructor) -> OpConstructor:
    """
    Constructs the controlled version of a given qubit operation
    :param op: operation to control
    :return: A Class C-Op which takes as a first input the controlling qubit and
    remaining inputs as a normal op.
    """
    if hasattr(op, 'wrap_op_hook'):
        return op.wrap_op_hook(C, consumed_inputs=[0]) or QubitOpWrapper(COp, op)
    else:
        return QubitOpWrapper(COp, op)


class COp(MatrixOp):
    """Contains special logic in the qip.ext.kronprod code"""

    def __init__(self, op: OpConstructor, *inputs: Qubit, **kwargs):
        if len(inputs) < 2:
            raise ValueError("Not enough input values given.")
        if inputs[0].n != 1:
            raise ValueError("Control bit can only be of size n=1")
        if len(set(inputs)) < len(inputs):
            raise ValueError("Cannot have repeated qubits in COp")

        # Directly editing kwargs causes issues on subsequent calls, and putting it in
        # the op call directory can cause issues with C(C(...)) calls with multiple nosink defs.
        nkwargs = kwargs.copy()
        nkwargs['nosink'] = True
        nkwargs['nosplit'] = True
        self.op = cast(MatrixOp, self.make_op(op, inputs, nkwargs))

        if not issubclass(type(self.op), MatrixOp):
            raise ValueError("C(Op) may only be applied to matrix ops and swaps, not {}".format(type(self.op)))

        super().__init__(*inputs, qid=self.op.qid, **kwargs)

    def make_op(self, op: OpConstructor, inputs: Sequence[Qubit], nkwargs: Mapping[str, Any]):
        return op(*inputs[1:], **nkwargs)

    def makemats(self, index_groups: Sequence[Sequence[int]]) -> Mapping[IndexType, MatrixType]:
        opm = self.op.makemats(index_groups[1:])
        newdict = {}
        for indices in opm:
            newindices = tuple(flatten([index_groups[0], indices]))
            newdict[newindices] = CMat(opm[indices])
        return newdict

    def __repr__(self):
        return "C{}".format(self.op)


class CMat(object):
    _kron_struct = 2

    def __init__(self, mat: Sequence[Sequence[complex]]):
        if type(mat) == list:
            self.m = numpy.array(mat)
        else:
            self.m = cast(numpy.ndarray, mat)
        self.shape = (self.m.shape[0]*2, self.m.shape[1]*2)

    def __getitem__(self, item: Union[int, Tuple[int, int]]):
        if type(item) == tuple and len(item) == 2:
            row, col = item[0], item[1]
            if row < self.shape[0]/2 and col < self.shape[1]/2:
                return 1.0 if row == col else 0.0
            elif row >= self.shape[0]/2 and col >= self.shape[1]/2:
                r, c = row - int(self.shape[0]/2), col - int(self.shape[1]/2)
                return self.m[r, c]
            else:
                return 0.0
        else:
            raise ValueError("CMat can only be indexed with M[i,j] not M[{}]".format(item))


class FOp(Qubit):
    """
    Operation to apply function to circuit. takes a function f and two registers: maps |x>|q> values to |x>|f(x)^q>
    """
    def __init__(self, func: Callable[[int], int], reg1: Qubit, reg2: Qubit, **kwargs):
        """
        Constructs FOp operator.
        :param func: function to apply to circuit
        :param reg1: register to take as input to function
        :param reg2: register to "store output", index xor'ed against function output.
        """
        super().__init__(reg1, reg2, **kwargs)
        self.func = func
        self.reg1 = reg1
        self.reg2 = reg2

    def feed_indices(self, inputvals: StateType, index_groups: Sequence[Sequence[int]], n: int, arena: StateType,
                     backend: Backend) -> Tuple[StateType, StateType, Tuple[int, int]]:
        reg1 = numpy.array(index_groups[0], dtype=numpy.int32)
        reg2 = numpy.array(index_groups[1], dtype=numpy.int32)
        backend.func_apply(reg1, reg2, self.func, inputvals, n, arena)
        return arena, inputvals, (0, 0)

    def __repr__(self) -> str:
        return "F({}, {})".format(self.reg1, self.reg2)


F = QubitOpWrapper(FOp)
