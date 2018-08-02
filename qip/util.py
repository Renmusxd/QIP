from __future__ import print_function

import numpy
import collections


def kronselect_dot(mats, vec, n, outputarray, input_offset=0, output_offset=0, dot_impl=None):
    """
    Efficiently performs the operation: OuterProduct( m1, m2, ..., mn ) dot vec
    for the case where most mj are identity.
    :param mats: { (indices,... #m) : mat(2**m x 2**m) }
    :param vec: vector v of size 2**n
    :param n: total number of matrices (including identities)
    :param outputarray: array in which to store output
    :param dot_impl: implementation of cdot function to use, see dot_loop below for example.
    :param input_offset: offset at which vec starts relative to possible larger context
    :param output_offset: offset at which outputarray starts relative to possible larger context
    """
    if len(vec) + input_offset > 2**n:
        raise ValueError("Input vector size plus offset may be no larger than the total number of qubit states (2^n)")

    if len(outputarray) + output_offset > 2**n:
        raise ValueError("Output vector size plus offset may be no larger than the total number of qubit states (2^n)")

    newmats = {}
    for indices in mats:
        # Will be set by one of the paths
        nindices = 0
        m = None

        if type(indices) != tuple and type(indices) != int:
            raise Exception("Type of indices must be tuple: {}".format(indices))
        elif type(indices) == tuple:
            m = mats[indices]
            if type(m) == list:
                m = numpy.array(m)
            newmats[indices] = m
            nindices = len(indices)
        elif type(indices) == int:
            m = mats[indices]
            if type(m) == list:
                m = numpy.array(m)
            newmats[(indices,)] = m
            nindices = 1

        if 2**nindices != m.shape[0] or 2**nindices != m.shape[1]:
            raise Exception("Shape of square submatrix must equal 2**(number of indices): "
                            "{}: {}".format(indices, m))

    if dot_impl is not None:
        iter_indices = newmats.keys()
        cindices = numpy.array([numpy.array(x, dtype=numpy.int32) for x in iter_indices])
        cmats = numpy.array([newmats[x] if type(newmats[x]) != numpy.ndarray else newmats[x].astype(numpy.complex128)
                             for x in iter_indices])
        dot_impl(cindices, cmats, vec, n, outputarray, input_offset=input_offset, output_offset=output_offset)
    else:
        dot_loop(newmats, vec, n, outputarray, input_offset=input_offset, output_offset=output_offset)


def dot_loop(mats, vec, n, output, input_offset=0, output_offset=0):
    allindices = list(mats.keys())
    flatindices = list(sorted(set(index for indices in allindices for index in indices)))

    for outputrow in range(len(output)):
        row = outputrow + output_offset
        s = 0
        for rowcol, mijs in gen_valid_col_and_matcol(row, flatindices, n):
            r, c = rowcol
            input_col = c - input_offset
            if input_col < 0 or input_col >= len(vec):
                continue

            p = 1.0
            # Multiply required entries in each non-indentity matrix
            for indices in allindices:
                mat = mats[indices]
                submati = bitarray_to_uint([mijs[index][0] for index in indices])
                submatj = bitarray_to_uint([mijs[index][1] for index in indices])
                p *= mat[submati, submatj]

            s += p*vec[input_col]
        output[outputrow] = s
    return output


def gen_valid_col_and_matcol(row, matindices, n):
    rowbits = uint_to_bitarray(row, n)
    colbits = rowbits[:]

    matrow = tuple(rowbits[indx] for indx in matindices)
    for i in range(2**len(matindices)):
        matcol = uint_to_bitarray(i, len(matindices))
        for j, indx in enumerate(matindices):
            colbits[indx] = matcol[j]
        yield (row, bitarray_to_uint(colbits)), {matindices[j]: item for j, item in enumerate(zip(matrow, matcol))}


def gen_edit_indices(index_groups, maxindex):
    if len(index_groups) > 0:
        allindices = flatten(index_groups)

        bits = [0] * (maxindex+1)
        for i in range(2**len(allindices)):
            flips = uint_to_bitarray(i, len(allindices))
            for j, bit in enumerate(flips):
                bits[allindices[j]] = flips[j]

            qbit_state_indices = [0] * len(index_groups)
            indx = 0
            for j, index_group in enumerate(index_groups):
                subflips = flips[indx:indx+len(index_group)]
                qbit_state_indices[j] = bitarray_to_uint(subflips)
                indx += len(index_group)
            yield bitarray_to_uint(bits), qbit_state_indices


def expand_kron_matrix(mats, n):
    m = numpy.zeros((2**n, 2**n), dtype=numpy.complex128)
    mats = {i: numpy.array(mats[i], dtype=numpy.complex128) for i in mats}
    for i in range(2**n):
        v = numpy.zeros((2**n,), dtype=numpy.complex128)
        v[i] = 1.0
        kronselect_dot(mats, v, n, m[i, :])
    return m


def uint_to_bitarray(num, n):
    bits = []
    for i in range(n):
        bits.append(num % 2)
        num = num >> 1
    return bits[::-1]


def bitarray_to_uint(bits):
    s = 0
    for i in range(len(bits)):
        s += 2**i if bits[len(bits)-i-1] else 0
    return s


def flatten(lst):
    listgen = [item if isinstance(item, collections.Iterable) else (item,) for item in lst]
    return [item for sublist in listgen for item in sublist]


def qubit_index_notation(i, *qns, n=None):
    if n is None:
        n = sum(qns)
    index_array = []
    bit_array = uint_to_bitarray(i, n)
    start_indx = 0
    for qubit_size in qns:
        num = bitarray_to_uint(bit_array[start_indx:start_indx + qubit_size])
        index_array.append(num)
        start_indx += qubit_size
    return index_array


def gen_qubit_prints(state, *qs):
    qubit_sizes = []
    for q in qs:
        if type(q) == int:
            qubit_sizes.append(q)
        else:
            qubit_sizes.append(q.n)
    n = sum(qubit_sizes)
    for i in range(len(state)):
        if state[i] == 0:
            continue
        s = "|"
        s += ",".join(map(str,qubit_index_notation(i,*qubit_sizes,n=n)))
        s += "> = {}".format(state[i])
        yield s

