from qip.distributed.proto import ComplexVector, ComplexMatrix, Indices
from typing import Union, Sequence, Iterable, Tuple
import numpy


def indices_to_pbindices(indices: Iterable[int]) -> Indices:
    pass


def pbindex_to_index(indices: Indices) -> Tuple[int, ...]:
    pass

def vec_to_pbvec(vec: Union[Sequence[complex], numpy.ndarray]) -> ComplexVector:
    pass


def pbvec_to_vec(vec: ComplexVector) -> numpy.ndarray:
    pass