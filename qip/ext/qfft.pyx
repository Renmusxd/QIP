from __future__ import division
import numpy as np
import cython
cimport numpy as np
import math
cimport util
from util cimport *


@cython.wraparound(False)
@cython.boundscheck(False)
def qfft(int[:] indices, int n, double complex[:] vec, double complex[:] output):
    cdef int i
    cdef int j
    cdef int m = indices.shape[0]
    cdef int vec_index
    cdef int base_inc
    cdef int[:] base_vec_indices = np.ndarray(shape=(2**m,), dtype=np.int32)
    cdef int[:] other_indices = np.array([i for i in np.arange(0,n) if i not in indices],dtype=np.int32)
    cdef double complex[:] qfft_buff = np.ndarray(shape=(2**m), dtype=np.complex128)
    cdef double complex[:] qfft_output #= np.ndarray(shape=(2**m), dtype=np.complex128)
    cdef double fft_normalizer = np.power(qfft_buff.shape[0],-0.5)

    with nogil:
        # Calculate the "base" indices to consider for each fft.
        for i in range(2**m):
            vec_index = 0
            for j in range(m):
                vec_index = set_bit(vec_index, (n-1) - indices[j],
                                    get_bit(i, (m-1) - j))
            base_vec_indices[i] = vec_index

        # For each set of ffts to perform
        for i in range(2**(n-m)):
            base_inc = 0
            for j in range(n-m):
                base_inc = set_bit(base_inc, (n-1) - other_indices[j],
                                   get_bit(i, ((n-m)-1) - j))

            # Fill qfft buffer from relevant indices.
            for j in range(2**m):
                qfft_buff[j] = vec[base_inc | base_vec_indices[j]]

            # Do fft
            # TODO find correct fft library to call
            with gil:
                qfft_output = np.fft.fft(qfft_buff)*fft_normalizer

            # Put back in place
            for j in range(2**m):
                output[base_inc | base_vec_indices[j]] = qfft_output[j]


@cython.wraparound(False)
@cython.boundscheck(False)
def qifft(int[:] indices, int n, double complex[:] vec, double complex[:] output):
    cdef int i
    cdef int j
    cdef int m = indices.shape[0]
    cdef int vec_index
    cdef int base_inc
    cdef int[:] base_vec_indices = np.ndarray(shape=(2**m,), dtype=np.int32)
    cdef int[:] other_indices = np.array([i for i in np.arange(0,n) if i not in indices],dtype=np.int32)
    cdef double complex[:] qifft_buff = np.ndarray(shape=(2**m), dtype=np.complex128)

    with nogil:
        # Calculate the "base" indices to consider for each fft.
        for i in range(2**m):
            vec_index = 0
            for j in range(m):
                vec_index = set_bit(vec_index, (n-1) - indices[j],
                                    get_bit(i, (m-1) - j))
            base_vec_indices[i] = vec_index

        # For each set of ffts to perform
        for i in range(2**(n-m)):
            base_inc = 0
            for j in range(n-m):
                base_inc = set_bit(base_inc, (n-1) - other_indices[j],
                                   get_bit(i, ((n-m)-1) - j))

            # Fill qfft buffer from relevant indices.
            for j in range(2**m):
                qifft_buff[j] = vec[base_inc | base_vec_indices[j]]

            # Do fft
            # TODO find correct ifft library to call
            with gil:
                qifft_buff = np.fft.ifft(qifft_buff)

            # Put back in place
            for j in range(2**m):
                output[base_inc | base_vec_indices[j]] = qifft_buff[j]