from qip.qip import Qubit
from qip.operators import H, C, Rm, Swap
from qip.qubit_util import QubitFuncWrapper
from qip.util import flatten
from typing import Union, Tuple, MutableSequence


@QubitFuncWrapper.wrap
def QFFT(*inputs: Qubit, rev: bool = True) -> Union[Qubit, Tuple[Qubit, ...]]:
    qarr = flatten([inp.split(range(inp.n)) for inp in inputs])
    recQFFT(qarr)

    if rev:
        half = int(len(qarr) / 2)
        for i in range(half):
            qarr[i], qarr[-1-i] = Swap(qarr[i], qarr[-1-i])

    if len(inputs) < len(qarr):
        outputqarr = []
        index = 0
        for qubit in inputs:
            outputqarr.append(Qubit(*(qarr[index:index+qubit.n])))
            index += qubit.n
    else:
        outputqarr = qarr

    if len(outputqarr) > 1:
        return tuple(outputqarr)
    else:
        return outputqarr[0]


def recQFFT(qarr: MutableSequence[Qubit], offset: int = 0):
    if len(qarr) > offset:
        qarr[offset] = H(qarr[offset])
        for i in range(offset+1, len(qarr)):
            m = 1 + i - offset
            qarr[i], qarr[offset] = C(makeR(m))(qarr[i], qarr[offset])
        recQFFT(qarr, offset + 1)


def makeR(m: int) -> Rm:
    return lambda *inputs, **kwargs: Rm(m, *inputs, negate=True, **kwargs)
