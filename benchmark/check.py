from qip.operators import *
from qip.pipeline import run
import time
from matplotlib import pyplot

def timef(f, *args, iters=1):
    acc = 0
    for i in range(iters):
        start = time.time()
        f(*args)
        acc += time.time() - start
    return acc / float(iters)

def test(n):
    q1 = Qubit(n=n-1)
    q2 = Qubit(n=1)
    n2 = Not(q2)
    o = run(q1, n2, feed={q2: [1.0, 0.0]})


ns = list(range(2,25,1))
times = [timef(test, n, iters=10) for n in ns]

pyplot.semilogy(ns,times)
pyplot.show()