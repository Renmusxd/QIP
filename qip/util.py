import numpy


def kronselect_dot(mats,vec,n):
    """
    Efficiently performs the operation: OuterProduct( m1, m2, ..., mn ) dot vec
    for the case where most mj are identity.
    :param mats: { (indices,...) : mat(2x2) }
    :param vec: vector v of size 2**n
    :param n: total number of matrices (including identities)
    :return:
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
    # Sort all keys and make into tuples
    mats = newmats

    output = numpy.zeros(shape=len(vec))

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
        for j,indx in enumerate(matindices):
            colbits[indx] = matcol[j]
        yield (row,bitarray_to_uint(colbits)), {matindices[j]: item for j, item in enumerate(zip(matrow, matcol))}


def gen_edit_indices(indices):
    if len(indices) > 0:
        indices = list(sorted(indices))
        maxindex = indices[-1]
        bits = [0] * (maxindex+1)
        for i in range(2**len(indices)):
            flips = uint_to_bitarray(i, len(indices))
            for j, bit in enumerate(flips):
                bits[indices[j]] = bit
            yield bitarray_to_uint(bits), flips


def uint_to_bitarray(num, n):
    bits = []
    for i in range(n):
        bits.append(num % 2)
        num = num >> 1
    return list(reversed(bits))


def bitarray_to_uint(bits):
    s = 0
    for i, bit in enumerate(reversed(bits)):
        s += 2**i if bit else 0
    return s


def flatten(list):
    return [item for sublist in list for item in sublist]