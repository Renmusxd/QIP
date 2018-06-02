import numpy as np
import cython
cimport numpy as np
cimport util
from util cimport *


@cython.wraparound(False)
@cython.boundscheck(False)
def func_apply(int[:] reg1_indices, int[:] reg2_indices, func,
              double complex[:] vec, int n, double complex[:] output):
    """
    Maps |x>|0> to |x>|func(x)> for |reg1>|reg2>
    Coefficients on all |x>|y!=0> must be |0>
    :param reg1_indices: indices for register 1
    :param reg2_indices: indices for register 2
    :param func: function to apply or string for pre-defined c functions.
    :param vec: input state
    :param n: total number of qubits
    :param output: output state (will be overwritten).
    """
    if vec.shape[0] != output.shape[0] or 2**n != vec.shape[0]:
        raise ValueError("Both input and output vectors must be of size 2**n")

    # Plan:
    # define |psi> = |x>|y>|r> where |r> is the state of qubits not in reg1 or reg2
    # Iterate through all |x>|0>|r> and copy value over to |x>|f(x)>|r>
    # Loop through |x> values, get |f(x)>, loop over all |r>

    # First zero the output to account for missed |y != f(x)> values
    cdef int i
    cdef int j
    with nogil:
        for i in range(2**n):
            output[i] = 0.0

    cdef int x
    cdef int y
    cdef int r
    cdef double complex tmp
    cdef int vec_zero_index
    cdef int vec_f_offset
    cdef int vec_r_offset
    cdef int n_remaining = n - (reg1_indices.shape[0] + reg2_indices.shape[0])
    cdef int[:] other_indices = np.array([i for i in np.arange(0,n)
                                          if (i not in reg1_indices and i not in reg2_indices)],
                                         dtype=np.int32)
    cdef int n_reg1 = reg1_indices.shape[0]
    cdef int n_reg2 = reg2_indices.shape[0]

    with nogil:
        for x in range(2**n_reg1):
            with gil:
                # Will only use last n_reg2 bytes.
                y = func(x)

            # Make base index |x>|0>
            # We are assuming that |x>|y!=0> is 0 for all the input.
            vec_zero_index = 0
            for j in range(reg1_indices.shape[0]):
                vec_zero_index = set_bit(vec_zero_index,
                                         (n-1) - reg1_indices[j],
                                         get_bit(x, (n_reg1-1) - j))
            # Make base index |x>|f(x)>
            vec_f_offset = 0
            for j in range(reg2_indices.shape[0]):
                vec_f_offset = set_bit(vec_f_offset,
                                       (n-1) - reg2_indices[j],
                                       get_bit(y, (n_reg2-1) - j))

            # Now can use to calculate offsets.
            # |x>|f(x)>|r> = |x>|0>|r> for all |r>
            if vec_f_offset != 0:
                for r in range(2**other_indices.shape[0]):
                    vec_r_offset = 0
                    # TODO check this code.
                    for j in range(reg2_indices.shape[0]):
                        vec_r_offset = set_bit(vec_r_offset, (n-1) - other_indices[j],
                                               get_bit(r, (n_remaining-1) - j))
                    output[vec_zero_index | vec_f_offset | vec_r_offset] = vec[vec_zero_index | vec_r_offset]