import numpy as np
cimport cython
from cython cimport parallel
cimport numpy as np
cimport util
from util cimport *


@cython.wraparound(False)
@cython.boundscheck(False)
def func_apply(int[:] reg1_indices, int[:] reg2_indices, func,
               double complex[:] vec, int n, double complex[:] output,
               int input_offset=0, int output_offset=0, pregen=True):
    """
    Maps |x>|q> to |x>|func(x) xor q> for |reg1>|reg2>
    :param reg1_indices: indices for register 1
    :param reg2_indices: indices for register 2
    :param func: function to apply or string for pre-defined c functions.
    :param vec: input state
    :param n: total number of qubits
    :param output: output state (will be overwritten).
    :param input_offset: offset at which input vec starts
    :param output_offset: offset at which output vec starts
    :param pregen: True to pregenerate function output to speed up loop, good for reg1.n ~= reg2.n
    """
    if vec.shape[0] != output.shape[0] or 2**n != vec.shape[0]:
        raise ValueError("Both input and output vectors must be of size 2**n")
    if len(vec) + input_offset > 2**n:
        raise ValueError("Input vector size plus offset may be no larger than the total number of qubit states (2^n)")
    if len(output) + output_offset > 2**n:
        raise ValueError("Output vector size plus offset may be no larger than the total number of qubit states (2^n)")
    # Plan:
    # define |psi> = |x>|y>|r> where |r> is the state of qubits not in reg1 or reg2
    # Iterate through all |x>|q>|r> and copy value over to |x>|f(x)^q>|r>
    # Loop through |x> values, get |f(x)>, loop over all |r>

    # First zero the output to account for missed |y != f(x)> values
    cdef:
        int i
        int j

    with nogil:
        for i in parallel.prange(2**n, schedule='static'):
            output[i] = 0.0

    cdef:
        int x
        int y
        int q
        int r
        int c_pregen = int(pregen)
        double complex tmp
        int vec_zero_index
        int vec_f_offset
        int vec_q_offset
        int vec_r_offset
        int n_remaining = n - (reg1_indices.shape[0] + reg2_indices.shape[0])
        int[:] other_indices = np.array([i for i in np.arange(0,n)
                                            if (i not in reg1_indices and i not in reg2_indices)],
                                         dtype=np.int32)
        int n_reg1 = reg1_indices.shape[0]
        int n_reg2 = reg2_indices.shape[0]
        int reg2_value
        int[:] f_reg1 = np.zeros(shape=(1,), dtype=np.int32)  # Dummy value prevent unbound memory error.

    if c_pregen:
        f_reg1 = np.zeros(shape=(2**n_reg1,), dtype=np.int32)
        for x in range(2**n_reg1):
            f_reg1[x] = func(x)

    with nogil:
        for x in parallel.prange(2**n_reg1):
            if c_pregen:
                y = f_reg1[x]
            else:
                with gil:
                    # Will only use last n_reg2 bytes.
                    y = func(x)

            # Make base index |x>|0>
            vec_zero_index = 0
            for j in range(reg1_indices.shape[0]):
                vec_zero_index = set_bit(vec_zero_index,
                                         (n-1) - reg1_indices[j],
                                         get_bit(x, (n_reg1-1) - j))

            # Go through each possible reg2 value.
            for q in range(2**n_reg2):
                # f(x) xor q, called y^q from now on
                reg2_value = y ^ q

                # Make base index |x>|y^q>
                vec_f_offset = 0
                for j in range(reg2_indices.shape[0]):
                    vec_f_offset = set_bit(vec_f_offset,
                                           (n-1) - reg2_indices[j],
                                           get_bit(reg2_value, (n_reg2-1) - j))
                # Make base index |x>|q>
                vec_q_offset = 0
                for j in range(reg2_indices.shape[0]):
                    vec_q_offset = set_bit(vec_q_offset,
                                           (n-1) - reg2_indices[j],
                                           get_bit(q, (n_reg2-1) - j))

                # Now can use to calculate offsets.
                # |x>|y^q>|r> = |x>|q>|r> for all |r>
                for r in range(2**other_indices.shape[0]):
                    vec_r_offset = 0
                    for j in range(reg2_indices.shape[0]):
                        vec_r_offset = set_bit(vec_r_offset, (n-1) - other_indices[j],
                                               get_bit(r, (n_remaining-1) - j))
                    output[vec_zero_index | vec_f_offset | vec_r_offset] = vec[vec_zero_index | vec_q_offset | vec_r_offset]
