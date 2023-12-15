import time
import math
import numpy as np
from gkdata import G_weights, K_weights, G_nodes, K_nodes
#=====================================================
# gkdata contains all the necessary nodes and weights
#=====================================================


def gausskronrod(f, a, b, args=(), set=(10, 21)):
    """
    Calculation of the integral of f
    with the Gauss-Kronrod quadrature method

    Parameters
    ----------
    f : callable
        function to integrate
    a : float
        beginning of the interval
    b : float
        end of the interval
    args : tuple, optional
        extra arguments to pass to f
    set : tuple, optional, default (G10, K21)
        possible set to use, the aviable are:
        (G7,  K15), (G10, K21), (G15, K31),
        (G20, K41), (G25, K51), (G30, K61)

    Return
    ----------
    I_K : float
        \int_a^b f
    error : float
        error of integration
    """

    N, M = set
    # b must be grather tha a
    if b < a:
        a, b = b, a
    else : pass

    mid = 0.5 * (b + a)
    dx  = 0.5 * (b - a)

    I_G = 0
    I_K = 0

    for gwi, gxi in zip(G_weights[N], G_nodes[N]):
        I_G += gwi*dx*f((gxi + 1)*dx + a, *args)

    for kwi, kxi in zip(K_weights[M], K_nodes[M]):
        I_K += kwi*dx*f((kxi + 1)*dx + a, *args)

    error = abs(I_G - I_K)

    return I_K, error


def adaptive_integrate(f, a, b, args=(), set=(10, 21), tol=1e-10):
    """
    Compute the integral of f; if tolerance is not satisfied,
    we split the interval in left and right part and recompute
    the integral and so on recursively until the tolerance is
    satisfied.

    Parameters
    ----------
    f : callable
        function to integrate
    a : float
        beginning of the interval
    b : float
        end of the interval
    args : tuple, optional
        extra arguments to pass to f
    set : tuple, optional, default (G10, K21)
        possible set to use, the aviable are:
        (G7,  K15), (G10, K21), (G15, K31),
        (G20, K41), (G25, K51), (G30, K61)
    tol : float, optional
        tollerance, default 1e-10

    Return
    ----------
    Int : float
        \int_a^b f
    err : float
        error on Int
    """
    # chek for integration
    val_g = [7,  10, 15, 20, 25, 30]
    val_k = [15, 21, 31, 41, 51, 61]
    L = [f'(G{g}, K{k})' for g, k in zip(val_g, val_k)]
    l = f'(G{set[0]}, K{set[1]})'

    if l not in L:
        raise Exception(f'Impossible to use {l}, you must choice between:\n{L}')

    # compute the integral
    Int, err = gausskronrod(f, a, b, args, set)
    # stop criteria
    if err < tol:
        return Int, err

    # mid point
    m = a + (b-a)/2
    # recursive calls
    I1, err1 = adaptive_integrate(f, a, m, args, set, tol)
    I2, err2 = adaptive_integrate(f, m, b, args, set, tol)

    Int = I1   + I2
    err = err1 + err2

    return Int, err


def test():
    """ little test
    """
    def h(x):
        """sine
        """
        return math.sin(x)
    def f(x, a1):
        """gaussian
        """
        return math.exp(-x**2/(2*a1))/(math.sqrt(2*math.pi)*a1)
    def g(x, a1, a2):
        """fermi dirac
        """
        return math.exp(-(x - a1)/a2)/(1 + math.exp(-(x - a1)/a2))

    I, dI = adaptive_integrate(h, 0, math.pi)
    print('Integral value is', I, '+-', dI)
    I, dI = adaptive_integrate(f, -100, 100, args=(1,))
    print('Integral value is', I, '+-', dI)
    I, dI = adaptive_integrate(g, 0, 20, args=(10, 0.02))
    print('Integral value is', I, '+-', dI)


if __name__ == '__main__':

    test()