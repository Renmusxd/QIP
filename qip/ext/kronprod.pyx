from __future__ import division
import numpy as np
import cython
cimport numpy as np
from libc.stdlib cimport malloc, free

# To remove dynamic dispatch, use the following enums/structs.
cdef enum MatrixType:
    OTHER, NUMPY_MAT, C_MAT, SWAP_MAT


cdef struct MatStruct:
    MatrixType mattype
    int param
    # A pointer to the original mat (except for CMAT)
    void* pointer


@cython.wraparound(False)
@cython.boundscheck(False)
def cdot_loop(int[:,:] indexgroups, matrices,
              double complex[:] vec, int n, double complex[:] output):
    # Check types
    if vec.shape[0] != output.shape[0] or 2**n != vec.shape[0]:
        raise ValueError("Both input and output vectors must be of size 2**n")

    # First convert to speedy data types
    # matrices should go from a list with objects of type (ndarray | C | Swap) to
    # an array of c structs with relevant information.
    cdef MatStruct* cmatrices = <MatStruct*>malloc(len(matrices)*sizeof(MatStruct))
    cdef MatStruct mstruct
    cdef np.ndarray[double complex,ndim=2,mode="c"] numpy_mat

    cdef int i
    for i in range(len(matrices)):
        mat = matrices[i]

        mstruct.pointer = <void*>mat

        struct_val = OTHER
        if hasattr(mat, '_kron_struct'):
            struct_val = mat._kron_struct
        elif type(mat) == np.ndarray:
            struct_val = NUMPY_MAT

        mstruct.mattype = struct_val
        if struct_val == NUMPY_MAT:
            numpy_mat = mat
            mstruct.pointer = <void*> numpy_mat.data
            mstruct.param = mat.shape[0]

        elif struct_val == SWAP_MAT:
            mstruct.param = mat.n

        elif struct_val == C_MAT:
            mstruct.param = mat.shape[0]
            mstruct.pointer = <void*>mat.m

        elif struct_val == OTHER:
            mstruct.param = mat.shape[0]

        cmatrices[i] = mstruct

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
                for j in range(nindices):
                    indx = flatindices[j]
                    # colbits[indx] = matcol[j]
                    colbits = set_bit(colbits, (n-1) - indx, get_bit(i, (nindices-1)-j))

                p = 1.0
                # Get entry in kron-matrix by multiplying relevant sub-matrices
                for indexgroupindex in range(nindexgroups):
                    # cdefed at c-typing speedup
                    indexgroup = indexgroups[indexgroupindex]
                    len_indexgroup = len(indexgroup)

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
    if mstruct.mattype == C_MAT:
        if mati < mstruct.param >> 1 and matj < mstruct.param >> 1:
            if mati == matj:
                return 1.0
            else:
                return 0.0
        elif mati >= mstruct.param >> 1 and matj >= mstruct.param >> 1:
            # In the future try recursively deducing the MatStruct types ahead of time.
            with gil:
                return (<object>mstruct.pointer)[mati - (mstruct.param >> 1), matj - (mstruct.param>>1)]
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


cdef int set_bit(int num, int bit_index, int value) nogil:
    return num ^ (-(value!=0) ^ num) & (1 << bit_index)


cdef int get_bit(int num, int bit_index) nogil:
    return (num >> bit_index) & 1