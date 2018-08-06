from qip.distributed.proto import ComplexVector, ComplexMatrix, Indices
from typing import Union, Sequence, Iterable, Tuple
import numpy


def indices_to_pbindices(indices: Iterable[int]) -> Indices:
    pbindices = Indices()
    for index in indices:
        pbindices.index.append(index)
    return pbindices


def pbindex_to_index(indices: Indices) -> Tuple[int, ...]:
    return tuple(index for index in indices.index)


def vec_to_pbvec(vec: Union[Sequence[complex], numpy.ndarray]) -> ComplexVector:
    pbvec = ComplexVector()
    for cval in vec:
        vec.real.append(cval.real)
        vec.imag.append(cval.imag)
    return pbvec


def pbvec_to_vec(vec: ComplexVector, dtype=numpy.complex128) -> numpy.ndarray:
    return numpy.fromiter(vec.real, dtype=dtype) + 1j*numpy.fromiter(vec.imag, dtype=dtype)
