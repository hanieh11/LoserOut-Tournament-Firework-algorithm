import numpy as np

#problem 1
def sphere(x):
    return np.sum(x**2)


def _w(v, c1, c2):
    return sum(c1 * np.cos(c2 * v))


# problem 9
def weierstrass(x):
    x = x + 0.5
    a, b, kmax = 0.5, 3.0, 20

    seq = np.array(range(kmax + 1))
    c1 = a**seq
    c2 = 2.0 * np.pi * (b**seq)
    return sum([_w(e, c1, c2) for e in x]) - _w(0.5, c1, c2) * len(x)


#problem 11
def rastrigin(x):
    return np.sum(x**2 - 10 * np.cos(2 * np.pi * x) + 10.0)


#problem 10
def griewank(x):
    d = len(x)
    m = np.multiply.reduce(np.cos(x / np.sqrt(range(1, d + 1))))
    s = sum(x**2 / 4000.0)
    return 1.0 + s - m


#problem 19
def ef8f2(xx):
    l = len(xx)
    x = 1 + xx
    y = 1 + np.insert(xx[1:l], l - 1, xx[0])

    f2 = 100.0 * (x * x - y)**2 + (1.0 - x)**2
    f = 1.0 + ((f2**2) / 4000.0 - np.cos(f2))
    return sum(f)

func_list = [sphere, rastrigin, griewank, ef8f2, weierstrass]