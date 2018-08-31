
cdef int entwine_bit(int n_indices, int n_non_indices, int v_indices, int v_non_indices, int mask) nogil:
    cdef int n = n_indices + n_non_indices
    cdef int indices_mask = 1 << (n_indices - 1)
    cdef int non_indices_mask = 1 << (n_non_indices - 1)
    cdef int bit_select = 1 << (n - 1)
    cdef int m = 0
    cdef int j
    # Push bits on from either non-indices or indices depending on mask
    for j in range(n):
        # push
        m = m << 1
        # select based on mask
        if (mask & bit_select) >> ((n-1) - j):
            n_indices -= 1
            m += (v_indices & indices_mask) >> n_indices
            indices_mask = indices_mask >> 1
        else:
            n_non_indices -= 1
            m += (v_non_indices & non_indices_mask) >> n_non_indices
            non_indices_mask = non_indices_mask >> 1
        bit_select = bit_select >> 1
    return m

cdef int set_bit(int num, int bit_index, int value) nogil:
    return num ^ (-(value!=0) ^ num) & (1 << bit_index)


cdef int get_bit(int num, int bit_index) nogil:
    return (num >> bit_index) & 1