# QIP
QIP is a python library for qubit simulation meant to feel like tensorflow or other graph pipeline libraries.
Users may define a set a qubits and then perform operations on them by constructing objects taking them as input. Since the library currently doesn't use a GPU no session object is required and the graphs may be easily reused.

## Example usage for CSwap inner product:
```python
# Define 11 qubits
q1 = Qubit(n=1)
q2 = Qubit(n=5)
q3 = Qubit(n=5)

# Perform a Hadamard transform on qubit 1
h1 = H(q1)

# Perform a CSwap operation on all 11 bits using the output
# of H(q1) as the control bit. Notice that Swap is its own
# operation and CSwap is constructed by applying C(...) to 
# the swap operation.
c1, c2, c3 = C(Swap)(h1, q2, q3)

# Finally perform another Hadamard transform on qubit 1
m1 = H(c1)

```
At this point we have constructed an object `m1` which is the output of the entire graph of operations. 
To run we must feed in initial values:
```python
state1 = [1.0, 0.0]

# Make two initial states for the pair of 5 qubit entries
state2 = numpy.zeros((32,))
state2[0] = 1.0

state3 = numpy.zeros((32,))
state3[1] = 1.0

# Feed the initial states for each set of qubits
o, _ = m1.run(feed={q1:state1, q2: state2, q3: state3})
```
`o` is a vector of size 2048 giving the complete state for each possible `|q1, q2, q3>`

## Installation
Installation via pip with `pip install qip` or manually by cloning the repo, install requirements from `requirements.txt` using `pip install -r requirements.txt`, and run `python setup.py build_ext --inplace && python setup.py install`.
