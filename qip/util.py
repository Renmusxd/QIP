import numpy
import collections
import qip.ext.kronprod as kronprod


def kronselect_dot(mats, vec, n, outputarray, cmode=True):
    """
    Efficiently performs the operation: OuterProduct( m1, m2, ..., mn ) dot vec
    for the case where most mj are identity.
    :param mats: { (indices,...) : mat(2x2) }
    :param vec: vector v of size 2**n
    :param n: total number of matrices (including identities)
    :param outputarray: array to store output in
    """
    if len(vec) != 2**n:
        raise Exception("Vec must be of length 2**n: {}:{}".format(len(vec), 2**n))

    newmats = {}
    for indices in mats:
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

    if cmode:
        iter_indices = newmats.keys()
        cindices = numpy.array([numpy.array(x) for x in iter_indices])
        cmats = numpy.array([newmats[x] for x in iter_indices])
        kronprod.cdot_loop(cindices, cmats, vec, n, outputarray)
    else:
        # Sort all keys and make into tuples
        dot_loop(newmats, vec, n, outputarray)


def dot_loop(mats, vec, n, output):
    allindices = list(mats.keys())
    flatindices = list(sorted(set(index for indices in allindices for index in indices)))

    for row in range(len(output)):
        s = 0
        for rowcol, mijs in gen_valid_col_and_matcol(row, flatindices, n):
            r, c = rowcol
            p = 1.0
            # Multiply required entries in each non-indentity matrix
            for indices in allindices:
                mat = mats[indices]
                submati = bitarray_to_uint([mijs[index][0] for index in indices])
                submatj = bitarray_to_uint([mijs[index][1] for index in indices])

                print(tuple([row, c, submati, submatj, mat[submati, submatj]]))
                p *= mat[submati, submatj]
            s += p*vec[c]
        output[row] = s
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


def gen_edit_indices(index_groups):
    if len(index_groups) > 0:
        allindices = flatten(index_groups)
        maxindex = max(allindices)

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


def expand_kron_matrix(mats, n, cmode=False):
    m = numpy.zeros((2**n, 2**n))
    for i in range(2**n):
        v = numpy.zeros((2**n,))
        v[i] = 1.0
        kronselect_dot(mats, v, n, m[i, :], cmode=cmode)
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


def flatten(list):
    listgen = [item if isinstance(item, collections.Iterable) else (item,) for item in list]
    return [item for sublist in listgen for item in sublist]
