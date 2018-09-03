"""
This code is admittedly a bit of a mess. Many things would be rewritten as functions if it didn't require mixing
python objects with c objects (i.e. structs).
"""
from __future__ import division

import numpy as np
cimport cython
from cython cimport parallel
cimport numpy as np
from libc.stdlib cimport malloc, free
import random
from util cimport *

# To remove dynamic dispatch, use the following enums/structs.
# if changed then change int values in operators.py
cdef enum MatrixType:
    OTHER, NUMPY_MAT, C_MAT, SWAP_MAT

cdef struct MatStruct:
    MatrixType mattype
    int param
    # A pointer to the original mat (except for CMAT)
    void* pointer


def count_extra_mats(mat):
    count = 0
    m = mat
    while hasattr(m, '_kron_struct'):
        struct_val = m._kron_struct
        if struct_val == C_MAT:
            m = m.m
            count += 1
        else:
            break
    return count


@cython.wraparound(False)
@cython.boundscheck(False)
def cdot_loop(int[:,:] indexgroups, matrices,
              double complex[:] vec, int n, double complex[:] output,
              int input_offset = 0, int output_offset = 0):
    """
    Efficiently compute the matrix multiplication of the matrix given by kron-product of the matrices each representing
    the indices in indexgroups with the vector in vec. Output is saved in output. input_offset and output_offset are
    used to compute subsections of the matrix multiplication by assuming vec[i] represents vec[i+offset] and similarly
    output[i] is output[i+offset].
    """
    # Get number of matrices including nested cmats
    cdef int n_extra_mats = 0
    for mat in matrices:
        n_extra_mats += count_extra_mats(mat)

    cdef:
        # First convert to speedy data types
        # matrices should go from a list with objects of type (ndarray | C | Swap) to
        # an array of c structs with relevant information.
        MatStruct* cmatrices = <MatStruct*>malloc((n_extra_mats+len(matrices))*sizeof(MatStruct))
        # Now add speedy types for nested C(...) values. Go as far down as possible and add pointers
        # to relevant addresses.

        MatStruct mstruct
        np.ndarray[double complex,ndim=2,mode="c"] numpy_mat

        int i
        int extra_i = len(matrices)
        # Bool was throwing errors with misdefined symbols
        int adding_extra_mats = False
        void** last_pointer_loc

    # A python list of numpy objects.
    numpy_objects = []

    for i in range(len(matrices)):
        mat = matrices[i]
        adding_extra_mats = False
        # Assign to null for loop invariant.
        last_pointer_loc = <void**>0

        # Recursively create mstructs for children of CMats
        while True:
            # Point to itself by default
            mstruct.pointer = <void*>mat

            # Find type of matrix
            struct_val = OTHER
            if hasattr(mat, '_kron_struct'):
                struct_val = mat._kron_struct

            elif type(mat) == np.ndarray:
                struct_val = NUMPY_MAT

            # Create struct from matrix
            mstruct.mattype = struct_val
            if struct_val == NUMPY_MAT:
                numpy_mat = np.ascontiguousarray(mat, np.complex128)
                numpy_objects.append(numpy_mat)
                # If numpy point to underlying data
                mstruct.pointer = <void*> numpy_mat.data
                mstruct.param = mat.shape[0]

            elif struct_val == SWAP_MAT:
                mstruct.param = mat.n

            elif struct_val == C_MAT:
                mstruct.param = mat.shape[0]
                # Points to python object of conditional operation
                # can be overwritten later with pointer to MatStruct
                mstruct.pointer = <void*>mat.m

            elif struct_val == OTHER:
                # Cannot use GIL objects in efficient loop for now.
                raise ValueError("Cannot pass matrices which are not numpy, SwapMat, or CMat")

            if not adding_extra_mats:
                cmatrices[i] = mstruct
                last_pointer_loc = &cmatrices[i].pointer
            else:
                cmatrices[extra_i] = mstruct
                last_pointer_loc[0] = &cmatrices[extra_i]
                last_pointer_loc = &cmatrices[extra_i].pointer
                extra_i += 1

            # We may need to repeat for cmats
            if struct_val != C_MAT:
                break
            else:
                mat = mat.m
                adding_extra_mats = True

    # Now put values into output
    cdef:
        double complex s, p, matentry
        int r, c
        int nindexgroups = len(indexgroups)
        long[:] indexgroupsizes = np.array([indexgroups[i].shape[0] for i in range(nindexgroups)], dtype=np.int)
        int[:] flatindices = np.array(list(sorted(set(index for indices in indexgroups for index in indices))),
                                           dtype=np.int32)
        int input_len = len(vec)
        int output_len = len(output)
        int nindices = len(flatindices)
        int two_nindices = 2**nindices
        int indx
        int colbits
        int submati
        int submatj
        int matindex
        int len_indexgroup
        int outputrow, row, vecrow, j, indexgroupindex, matindexindex

    # Entry the main loop.
    with nogil:
        # Build each entry in output.
        for outputrow in parallel.prange(output_len, schedule='static'):
            row = outputrow + output_offset
            colbits = row
            # Output accumulator
            s = 0.0

            # Generate valid columns and calculate required bitflips
            for i in range(two_nindices):
                for j in range(nindices):
                    indx = flatindices[j]
                    # colbits[indx] = matcol[j]
                    colbits = set_bit(colbits, (n-1) - indx, get_bit(i, (nindices-1)-j))

                vecrow = colbits - input_offset
                if vecrow < 0 or vecrow >= input_len:
                    continue

                # Get entry in kron-matrix by multiplying relevant sub-matrices
                p = 1.0
                for indexgroupindex in range(nindexgroups):
                    len_indexgroup = indexgroupsizes[indexgroupindex]
                    mstruct = cmatrices[indexgroupindex]

                    submati = 0
                    submatj = 0

                    # Get indices for matrix in mstruct (submati, submatj)
                    for matindexindex in range(len_indexgroup):
                        matindex = indexgroups[indexgroupindex,matindexindex]
                        submati = set_bit(submati, (len_indexgroup-1) - matindexindex,
                                          get_bit(row, (n-1) - matindex))
                        submatj = set_bit(submatj, (len_indexgroup-1) - matindexindex,
                                          get_bit(colbits, (n-1) - matindex))
                    matentry = calc_mat_entry(mstruct, submati, submatj)
                    p = p*matentry

                    if p == 0.0:
                        break

                s = s + (p*vec[vecrow])
            output[outputrow] = s

    # Finally.
    free(cmatrices)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double complex calc_mat_entry(MatStruct mstruct, int mati, int matj) nogil:
    """
    Calculate the value of the element at mstruct[mati,matj]
    :param mstruct: struct containing matrix information
    :param mati: row index
    :param matj: column index
    :return: mstruct[mati,matj]
    """
    cdef double complex* mmat
    cdef MatStruct* cmstruct
    if mstruct.mattype == C_MAT:
        if mati < mstruct.param >> 1 and matj < mstruct.param >> 1:
            if mati == matj:
                return 1.0
            else:
                return 0.0
        elif mati >= mstruct.param >> 1 and matj >= mstruct.param >> 1:
           cmstruct = <MatStruct*>mstruct.pointer
           return calc_mat_entry(cmstruct[0], mati - (mstruct.param >> 1), matj - (mstruct.param>>1))
        else:
            return 0.0

    elif mstruct.mattype == SWAP_MAT:
        if (mati & ~((-1) << mstruct.param)) == (matj >> mstruct.param) and\
                (matj & ~((-1) << mstruct.param)) == (mati >> mstruct.param):
            return 1.0
        return 0.0

    elif mstruct.mattype == NUMPY_MAT:
        mmat = <double complex*>mstruct.pointer
        return mmat[mati*mstruct.param + matj]


@cython.boundscheck(False)
@cython.wraparound(False)
def measure_probabilities(int[:] indices, int n, double complex[:] vec):
    measure_p = np.zeros(shape=(2**len(indices),), dtype=np.float64)
    cdef:
        double[:] p = measure_p
        int len_indices = indices.shape[0]
        int iter_num = 2**n
        int i, j
        int p_index = 0
        double complex vec_val
    with nogil:
        # Iterating over larger array in order likely better for cache.
        for i in range(iter_num):
            vec_val = vec[i]
            if vec_val == 0.0:
                continue
            p_index = entwine_bit(len_indices, 0, i, 0, -1)
            p[p_index] = p[p_index] + abs(vec_val)**2
    return measure_p


@cython.boundscheck(False)
@cython.wraparound(False)
def prob_magnitude(double complex[:] vec):
    cdef int i
    cdef double acc
    with nogil:
        for i in range(vec.shape[0]):
            acc += abs(vec[i])**2
    return acc


@cython.boundscheck(False)
@cython.wraparound(False)
def soft_measure(int[:] indices, int n, double complex[:] vec, measured=None):
    cdef:
        int iter_num = 2**vec.shape[0]
        int len_indices = indices.shape[0]
        int len_remaining_indices = n - len_indices
        int n_measured_indices = 2**len_indices
        int n_unmeasured_indices = 2**len_remaining_indices
        double[:] measure_probs
        double r, acc
        int i, j, p_index, p_index_index
        int m = 2**indices.shape[0] - 1
        int[:] to_measure
        int row_mask = 0

    if measured is None:
        to_measure = np.arange(0, n_measured_indices, dtype=np.int32)
    else:
        to_measure = np.arange(measured, measured+1, dtype=np.int32)

    r = random.random()
    # If only looking at subsection of wavefunction
    if indices.shape[0] < 2**n:
        r = r * prob_magnitude(vec)

    with nogil:
        for i in range(len_indices):
            row_mask = set_bit(row_mask, (n-1) - indices[i], 1)

        for p_index_index in range(to_measure.shape[0]):
            p_index = to_measure[p_index_index]

            acc = 0
            for i in range(n_unmeasured_indices):
                vec_index = entwine_bit(len_indices, len_remaining_indices, p_index, i, row_mask)
                acc += abs(vec[vec_index])**2

            r -= acc
            if r <= 0:
                # p_index and acc are set to correct values
                # can't return here because they aren't python objects and we are in nogil
                break
        # If finished loop without break then variables are set to last p_index and last acc -> correct output.
    return p_index, acc


@cython.boundscheck(False)
@cython.wraparound(False)
def measure(int[:] indices, int n, double complex[:] vec, double complex[:] out, measured=None, measured_prob=None):
    if measured is not None:
        if not (0 <= measured < 2**indices.shape[0]):
            raise ValueError("Measured value must be less than 2**len(indices)")
    if measured_prob is not None:
        if not 0.0 < measured_prob <= 1.0:
            raise ValueError("measured_prob must be 0 < p <= 1")

    cdef:
        int m
        double p
        int i
        int n_indices = indices.shape[0]
        int n_non_indices = n - n_indices
        int n_out = 2**(n - n_indices)
        int vec_row
        int row_mask = 0
        int indices_inc, non_indices_inc
        double mult_p

    if measured is None or measured_prob is None:
        m, p = soft_measure(indices, n, vec, measured=measured)
    else:
        m, p = measured, measured_prob
    mult_p = np.sqrt(1.0 / p)

    with nogil:
        for i in range(n_indices):
            row_mask = set_bit(row_mask, (n-1) - indices[i], 1)
        for i in range(n_out):
            vec_row = entwine_bit(n_indices, n_non_indices, m, i, row_mask)
            out[i] = vec[vec_row] * mult_p
    return m, p
