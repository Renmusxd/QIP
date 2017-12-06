from __future__ import division
import qip.operators as operators
import numpy as np

# "cimport" is used to import special compile-time information
# about the numpy module (this is stored in a file numpy.pxd which is
# currently part of the Cython distribution).
cimport numpy as np

from libc.stdlib cimport malloc, free

#from cython.parallel import prange, parallel
# with nogil, parallel(num_threads=8):
#   for i in prange(n, schedule='dynamic'):
#     ....


# We now need to fix a datatype for our arrays. I've used the variable
# DTYPE for this, which is assigned to the usual NumPy runtime
# type info object.
DTYPE = np.complex128
# "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix.
ctypedef np.complex128_t DTYPE_t

# To remove dynamic dispatch, use the following enums/structs.
cdef enum MatrixType:
    NUMPY_MAT, C_MAT, SWAP_MAT, OTHER

cdef struct MatStruct:
    MatrixType mattype
    int param
    # A pointer to the original mat (except for CMAT)
    void* pointer


def cdot_loop(np.ndarray indexgroups, matrices,
              double complex[:] vec, int n, double complex[:] output):
    # Check types
    if vec.shape[0] != output.shape[0] or 2**n != vec.shape[0]:
        raise ValueError("Both input and output vectors must be of size 2**n")

    # First convert to speedy data types
    # matrices should go from a list with objects of type (ndarray | C | Swap) to
    # an array of c structs with relevant information.
    cdef MatStruct* cmatrices = <MatStruct*>malloc(len(matrices)*sizeof(MatStruct))
    cdef np.ndarray[np.int_t, ndim=1] indexgroup
    cdef MatStruct mstruct
    cdef np.ndarray[double complex,ndim=2,mode="c"] numpy_mat

    cdef i
    for i in range(len(matrices)):
        mat = matrices[i]

        mstruct.pointer = <void*>mat

        if type(mat) == np.ndarray:
            mstruct.mattype = NUMPY_MAT
            numpy_mat = mat
            mstruct.pointer = <void*> numpy_mat.data
            mstruct.param = mat.shape[0]
        elif type(mat) == operators.SwapMat:
            mstruct.mattype = SWAP_MAT
            mstruct.param = mat.n
        elif type(mat) == operators.CMat:
            mstruct.mattype = C_MAT
            mstruct.param = mat.shape[0]
            mstruct.pointer = <void*>mat.m
        else:
            mstruct.mattype = OTHER
            mstruct.param = mat.shape[0]

        cmatrices[i] = mstruct

    # Now put values into output
    cdef DTYPE_t s, p
    cdef int r, c
    cdef int nindexgroups = len(indexgroups)
    #cdef np.ndarray[np.int_t, ndim=1] flatindices
    cdef long[:] flatindices
    flatindices = np.array(list(sorted(set(index for indices in indexgroups for index in indices))))
    cdef int nindices = len(flatindices)

    cdef int indx
    cdef int colbits
    cdef int submati
    cdef int submatj
    cdef int matindex
    cdef int len_indexgroup
    cdef int row, j, indexgroupindex, matindexindex

    # For each entry in the 2**n states (parallelize?)
    for row in range(2**n):
        s = 0.0
        colbits = row

        # Generate valid columns and calculate required bitflips
        for i in range(2**nindices):
            # Edit colbits
            for j in range(nindices):
                indx = flatindices[j]
                #colbits[indx] = matcol[j]
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
                #print(row,colbits,submati,submatj,matentry)
                p *= matentry

                # If p == 0.0 then exit early
                if p == 0.0:
                    break

            s += p*vec[colbits]
        output[row] = s

    with nogil:
        free(cmatrices)

    return output


cdef double complex calc_mat_entry(MatStruct mstruct, int mati, int matj):
    cdef object mobj
    cdef double complex* mmat
    if mstruct.mattype == C_MAT:
        if mati < mstruct.param/2 and matj < mstruct.param/2:
            if mati == matj:
                return 1.0
            else:
                return 0.0
        elif mati >= mstruct.param/2 and matj >= mstruct.param/2:
            mobj = (<object>mstruct.pointer)
            return mobj[mati - (mstruct.param >> 1), matj - (mstruct.param>>1)]
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
        mobj = (<object>mstruct.pointer)
        return mobj[mati,matj]

    else:
        raise ValueError("Struct not a valid type: "+str(mstruct.mattype))


cdef int set_bit(int num, int bit_index, int value):
    return num ^ (-(value!=0) ^ num) & (1 << bit_index)

cdef int get_bit(int num, int bit_index):
    return (num >> bit_index) & 1