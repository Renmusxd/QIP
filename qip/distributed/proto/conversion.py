from qip.operators import SwapMat, CMat
from qip.distributed.proto import *
from typing import Union, Sequence, Iterable, Tuple, Optional, Type, cast
import numpy


def indices_to_pbindices(indices: Iterable[int]) -> Indices:
    pbindices = Indices()
    for index in indices:
        pbindices.index.append(index)
    return pbindices


def pbindex_to_index(indices: Indices) -> Tuple[int, ...]:
    return tuple(index for index in indices.index)


def vec_to_pbvec(vec: Union[Sequence[complex], numpy.ndarray], pbvec: ComplexVector = None) -> ComplexVector:
    if pbvec is None:
        pbvec = ComplexVector()
    for cval in vec:
        pbvec.real.append(cval.real)
        pbvec.imag.append(cval.imag)
    return pbvec


def pbvec_to_vec(vec: ComplexVector, dtype=numpy.complex128) -> numpy.ndarray:
    return numpy.fromiter(vec.real, dtype=dtype) + 1j*numpy.fromiter(vec.imag, dtype=dtype)


def mat_to_pbmat(mat: Union[Sequence[Sequence[complex]], numpy.ndarray]) -> ComplexMatrix:
    pbmat = ComplexMatrix()
    numpymat = numpy.asarray(mat, dtype=numpy.complex128)
    for shape_val in numpymat.shape:
        pbmat.shape.append(shape_val)

    flatmat = numpymat.reshape((-1,))
    vec_to_pbvec(flatmat, pbvec=pbmat.data)
    return pbmat


def pbmat_to_mat(mat: ComplexMatrix, dtype=numpy.complex128) -> numpy.ndarray:
    return pbvec_to_vec(mat.data, dtype=dtype).reshape(tuple(mat.shape))


def matop_to_pbmatop(indices: Iterable[int], mat: Union[Sequence[Sequence[complex]], numpy.ndarray],
                     matop: Optional[MatrixOp] = None) -> MatrixOp:
    if matop is None:
        matop = MatrixOp()
    matop.indices.CopyFrom(indices_to_pbindices(indices))
    if isinstance(mat, SwapMat):
        matop.swap = True
    elif isinstance(mat, CMat):
        cmat = cast(CMat, mat)
        matop_to_pbmatop(indices[1:], cmat.m, matop.controlled_op)
    else:
        matop.matrix.CopyFrom(mat_to_pbmat(mat))
    return matop


def pbmatop_to_matop(pbmatop: MatrixOp) -> Tuple[Tuple[int, ...], object]:
    indices = pbindex_to_index(pbmatop.indices)
    if pbmatop.HasField('swap'):
        return indices, SwapMat(int(len(indices)/2))
    elif pbmatop.HasField('matrix'):
        return indices, pbmat_to_mat(pbmatop.matrix)
    elif pbmatop.HasField('controlled_op'):
        mat = pbmatop_to_matop(pbmatop.controlled_op)[1]
        return indices, CMat(mat)


def pbstatetype_to_statetype(pbstatetype: WorkerSetup.StateType) -> Type[numpy.complex_]:
    if pbstatetype == WorkerSetup.COMPLEX128:
        return numpy.complex128
    raise ValueError("Unknown enum value: {}".format(pbstatetype))