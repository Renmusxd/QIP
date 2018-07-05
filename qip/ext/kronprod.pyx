"""
This code is admittedly a bit of a mess. Many things would be rewritten as functions if it didn't require mixing
python objects with c objects (i.e. structs).
"""

from __future__ import division
import numpy as np
import cython
cimport numpy as np
from libc.stdlib cimport malloc, free
import math
import random
cimport util
from util cimport *

# To remove dynamic dispatch, use the following enums/structs.
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
              double complex[:] vec, int n, double complex[:] output):
    # Check types
    if vec.shape[0] != output.shape[0] or 2**n != vec.shape[0]:
        raise ValueError("Both input and output vectors must be of size 2**n")

    # Get number of matrices including nested cmats
    cdef n_extra_mats = 0
    for mat in matrices:
        n_extra_mats += count_extra_mats(mat)

    # First convert to speedy data types
    # matrices should go from a list with objects of type (ndarray | C | Swap) to
    # an array of c structs with relevant information.
    cdef MatStruct* cmatrices = <MatStruct*>malloc(len(matrices)*sizeof(MatStruct))
    # nullptr unless needed
    cdef MatStruct* extra_cmatrices
    if n_extra_mats:
       extra_cmatrices = <MatStruct*>malloc(n_extra_mats*sizeof(MatStruct))

    cdef MatStruct mstruct
    cdef np.ndarray[double complex,ndim=2,mode="c"] numpy_mat

    cdef int i, extra_i
    # Bool was throwing errors with misdefined symbols
    cdef int adding_extra_mats = False
    cdef void** last_pointer_loc
    for i in range(len(matrices)):
        mat = matrices[i]
        adding_extra_mats = False
        # Assign to null for loop invariant.
        last_pointer_loc = <void**>0

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
                mstruct.param = mat.shape[0]

            if not adding_extra_mats:
                cmatrices[i] = mstruct
                last_pointer_loc = &cmatrices[i].pointer
            else:
                extra_cmatrices[extra_i] = mstruct
                last_pointer_loc[0] = &extra_cmatrices[extra_i]
                last_pointer_loc = &extra_cmatrices[extra_i].pointer

            # We may need to repeat for cmats
            if struct_val != C_MAT:
                break
            else:
                mat = mat.m
                adding_extra_mats = True

    # Now put values into output
    cdef double complex s, p, matentry
    cdef int r, c
    cdef int nindexgroups = len(indexgroups)
    cdef int[:] flatindices = np.array(list(sorted(set(index for indices in indexgroups for index in indices))),
                                       dtype=np.int32)

    cdef int two_n = 2**n
    cdef int nindices = len(flatindices)
    cdef int two_nindices = 2**nindices
    cdef int indx
    cdef int colbits
    cdef int submati
    cdef int submatj
    cdef int matindex
    cdef int len_indexgroup
    cdef int row, j, indexgroupindex, matindexindex

    # This speeds up by 25%
    cdef int[:] indexgroup

    # Currently parallelism slows down because we can't take advantage
    # of the memory view long[:] up above.
    with nogil:
        for row in range(two_n):
            s = 0.0
            colbits = row

            # Generate valid columns and calculate required bitflips
            for i in range(two_nindices):
                # Edit colbits
                # TODO add system for getting "next non-zero index on row" for matrices like SWAP which are very sparse.
                for j in range(nindices):
                    indx = flatindices[j]
                    # colbits[indx] = matcol[j]
                    colbits = set_bit(colbits, (n-1) - indx, get_bit(i, (nindices-1)-j))

                p = 1.0
                # Get entry in kron-matrix by multiplying relevant sub-matrices
                for indexgroupindex in range(nindexgroups):
                    # For some reason it's still performing wrap-around and
                    # bounds checking here.
                    indexgroup = indexgroups[indexgroupindex,:]
                    len_indexgroup = indexgroup.shape[0]

                    mstruct = cmatrices[indexgroupindex]

                    submati = 0
                    submatj = 0

                    # Get indices for matrix in mstruct (submati, submatj)
                    for matindexindex in range(len_indexgroup):
                        matindex = indexgroup[matindexindex]
                        submati = set_bit(submati, (len_indexgroup-1) - matindexindex,
                                          get_bit(row, (n-1) - matindex))
                        submatj = set_bit(submatj, (len_indexgroup-1) - matindexindex,
                                          get_bit(colbits, (n-1) - matindex))
                    matentry = calc_mat_entry(mstruct, submati, submatj)
                    p = p*matentry

                    # If p == 0.0 then exit early
                    if p == 0.0:
                        break

                s = s + (p*vec[colbits])
            output[row] = s
        # nogil
        free(cmatrices)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double complex calc_mat_entry(MatStruct mstruct, int mati, int matj) nogil:
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

    elif mstruct.mattype == OTHER:
        with gil:
            return (<object>mstruct.pointer)[mati,matj]
    else:
        with gil:
            raise ValueError("Struct not a valid type: "+str(mstruct.mattype))


@cython.boundscheck(False)
@cython.wraparound(False)
def measure_probabilities(int[:] indices, int n, double complex[:] vec):
    measure_p = np.zeros(shape=(2**len(indices),), dtype=np.float64)
    cdef double[:] p = measure_p

    cdef int len_indices = indices.shape[0]
    cdef int iter_num = 2**n
    cdef int i, j
    cdef int p_index = 0
    cdef double complex vec_val
    with nogil:
        for i in range(iter_num):
            vec_val = vec[i]
            if vec_val == 0.0:
                continue

            for j in range(len_indices):
                # index = indices[j]
                p_index = set_bit(p_index, (len_indices-1) - j,
                                  get_bit(i, (n-1) - indices[j]))
            p[p_index] = p[p_index] + abs(vec_val.conjugate() * vec_val)
    return measure_p


@cython.boundscheck(False)
@cython.wraparound(False)
def measure(int[:] indices, int n, double complex[:] vec, double complex[:] out, measured=None, measured_prob=None):
    if measured is not None:
        if not (0 <= measured < 2**indices.shape[0]):
            raise ValueError("Measured value must be less than 2**len(indices)")
    if measured_prob is not None:
        if not 0.0 < measured_prob <= 1.0:
            raise ValueError("measured_prob must be 0 < p <= 1")

    cdef int i
    cdef double r
    if measured_prob is None or measured is None:
        measure_probs = measure_probabilities(indices, n, vec)
        if measured is None:
            r = random.random()
            for i in range(2**indices.shape[0]):
                r -= measure_probs[i]
                if r <= 0:
                    break
            measured = i
        measured_prob = measure_probs[measured]

    cdef int m = measured
    cdef double p = measured_prob
    cdef double mult_p = math.sqrt(1.0 / p)

    cdef int n_indices = indices.shape[0]
    cdef int n_non_indices = n - n_indices
    cdef int n_out = 2**(n - n_indices)
    cdef int vec_row
    cdef int row_mask = 0
    cdef int indices_inc, non_indices_inc
    with nogil:
        for i in range(n_indices):
            row_mask = set_bit(row_mask, (n-1) - indices[i], 1)
        for i in range(n_out):
            vec_row = entwine_bit(n_indices, n_non_indices, m, i, row_mask)
            out[i] = vec[vec_row] * mult_p
    return m, p
