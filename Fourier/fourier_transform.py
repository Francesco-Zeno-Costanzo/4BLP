"""
Implementetion of DFT and FFT
"""
import time
import numpy as np
import matplotlib.pyplot as plt


def DFT(x, anti=-1):
    '''
    Compute the discrete Fourier Transform of the 1D array x

    Parameters
    ----------
    x : 1darray
        data to transform
    anti : int, optional
        -1 trasform
         1 anti trasform

    Return
    ------
    dft : 1d array
        dft or anti dft of x
    '''

    N = len(x)        # length of array
    n = np.arange(N)  # array from 0 to N
    k = n[:, None]    # transposed of n written as a Nx1 matrix
    # is equivalent to k = np.reshape(n, (N, 1))
    # so k * n will be a N x N matrix

    M = np.exp(anti * 2j * np.pi * k * n / N)
    dft = M @ x

    if anti == 1:
        return dft/N
    else:
        return dft


def FFT(x, anti=-1):
    '''
    Compute the Fast Fourier Transform of the 1D array x.
    Using non recursive Cooley-Tukey FFT.
    In recursive FFT implementation, at the lowest
    recursion level we must perform  N/N_min DFT.
    The efficiency of the algorithm would benefit by
    computing these matrix-vector products all at once
    as a single matrix-matrix product.
    At each level of recursion, we also perform
    duplicate operations which can be vectorized.

    Parameters
    ----------
    x : 1darray
        data to transform
    anti : int, optional
        -1 trasform
         1 anti trasform

    Return
    ------
    fft : 1d array
        fft or anti fft of x

    '''
    N = len(x)

    if np.log2(N) % 1 > 0:
        msg_err = "The size of x must be a apower of 2"
        raise ValueError(msg_err)

    # stop criterion
    N_min = min(N, 2**2)

    #  DFT on all length-N_min sub-problems
    n = np.arange(N_min)
    k = n[:, None]
    M = np.exp(anti * 2j * np.pi * n * k / N_min)
    X = np.dot(M, x.reshape((N_min, -1)))

    while X.shape[0] < N:
        # first  part of the matrix, the one on the left
        X_even = X[:,:X.shape[1] // 2 ] # all rows, first  X.shape[1]//2 columns
        # second part of the matrix, the one on the right
        X_odd  = X[:, X.shape[1] // 2:] # all rows, second X.shape[1]//2 columns

        f = np.exp(anti * 1j * np.pi * np.arange(X.shape[0]) / X.shape[0])[:, None]
        X = np.vstack([X_even + f*X_odd, X_even - f*X_odd]) # re-merge the matrix

    fft = X.ravel() # flattens the array
    # from  matrix Nx1 to array with length N

    if anti == 1:
        return fft/N
    else :
        return fft


def RFFT(x, anti=-1):
    '''
    Compute the fft for real value using FFT
    only values corresponding to positive
    frequencies are returned.

    The transform is implemented by passing to
    a complex variable z = x[2n] + j x[2n+1]
    then an fft of length N/2 is calculated
    For the inverse we adopt an other method
    (the previous method didn't work,
    if you know how to fix it ... I would be grateful)

    Parameters
    ----------
    x : 1darray
        data to transform
    anti : int, optional
        -1 trasform
         1 anti trasform

    Return
    ------
    rfft : 1d array
        rfft or anti rfft of x
    '''
    if anti == -1 :
        z  = x[0::2] + 1j * x[1::2]  # Splitting odd and even
        Zf = FFT(z)
        Zc = np.array([Zf[-k] for k in range(len(z))]).conj()
        Zx =  0.5  * (Zf + Zc)
        Zy = -0.5j * (Zf - Zc)

        N = len(x)
        W = np.exp(- 2j * np.pi * np.arange(N//2) / N)
        Z = np.concatenate([Zx + W*Zy, Zx - W*Zy])

        return Z[:N//2+1]

    if anti == 1 :
        # we use the fft symmetries to reconstruct the whole spectrum
        N = 2*(len(x)-1) # length of final array
        x1 = x[:-1]      # cut last value
        S = len(x1)      # length of new array
        xn = np.zeros(N, dtype=complex)
        xn[0:S] = x1
        xx = x[1:]       # cut first element, zero frequency mode
        xx = xx[::-1]    # rewind array
        xn[S:N] = xx.conj()
        z = FFT(xn, anti=1)

        return np.real(z)


def fft_freq(n, d, real):
    '''
    Return the Discrete Fourier Transform sample frequencies.
    if real = False then:
    f = [0, 1, ...,   n/2-1,     -n/2, ..., -1] / (d*n)   if n is even
    f = [0, 1, ..., (n-1)/2, -(n-1)/2, ..., -1] / (d*n)   if n is odd
    else :
    f = [0, 1, ...,     n/2-1,     n/2] / (d*n)   if n is even
    f = [0, 1, ..., (n-1)/2-1, (n-1)/2] / (d*n)   if n is odd

    Parameters
    ----------
    n : int
        length of array that you transform

    d : float
        Sample spacing (inverse of the sampling rate).
        If the data array is in seconds
        the frequencies will be in hertz
    real : bool
        false for fft
        true for rfft

    Returns
    -------
    f: 1d array
        Array of length n containing the sample frequencies.
    '''
    if not real:
        if n%2 == 0:
            f1 = np.array([i for i in range(0, n//2)])
            f2 = np.array([i for i in range(-n//2,0)])
            return np.concatenate((f1, f2))/(d*n)
        else :
            f1 = np.array([i for i in range((n-1)//2 + 1)])
            f2 = np.array([i for i in range(-(n-1)//2, 0)])
            return np.concatenate((f1, f2))/(d*n)
    if real:
        if n%2 == 0:
            f1 = np.array([i for i in range(0, n//2 +1)])
            return f1 / (d*n)
        else :
            f1 = np.array([i for i in range((n-1)//2 +1)])
            return f1 / (d*n)


if __name__ == '__main__':

    # data
    x = np.linspace(0, 10, 2**11) # no more 2**14 for DFT
    y = 1 + 3*np.sin(2*(2*np.pi)*x) + np.sin(5*(2*np.pi)*x) + 0.5*np.sin(8*(2*np.pi)*x) + 0.001*np.sin(40*(2*np.pi)*x)
    noise = np.array([np.random.random() for _ in x])
    noise = 2 * noise - 1 # from [0, 1] to [-1, 1]
    intensity = 0.0
    y = y + intensity * noise
    #y = y - np.mean(y)
    #y = y*np.sin(40*(2*np.pi)*x)

    # DFT
    t0 = time.time()
    dft_m_i  = DFT(y)
    anti_dft = np.real(DFT(dft_m_i, anti=1))
    dt = time.time() - t0
    print(dt/2)

    #FFT
    t0 = time.time()
    fft_m_i  = FFT(y)
    anti_fft = np.real(FFT(fft_m_i, anti=1))
    dt = time.time()-t0
    print(dt/2)

    #RFFT
    t0 = time.time()
    fft_m_r   = RFFT(y)
    anti_rfft = np.real(RFFT(fft_m_r, anti=1))
    dt = time.time()-t0
    print(dt/2)

    #numpy FFT
    t0 = time.time()
    fft_n_i = np.fft.fft(y)
    fft_n_r = np.fft.rfft(y)
    dt = time.time()-t0
    print(dt/2)

    freq_m_i = fft_freq(len(y), x[-1]/len(y), real=False)
    freq_n_i = np.fft.fftfreq(len(y), x[-1]/len(y))
    freq_m_r = fft_freq(len(y), x[-1]/len(y), real=True)
    freq_n_r = np.fft.rfftfreq(len(y), x[-1]/len(y))

    ##Plot

    plt.figure(1)
    plt.title('Original Signal', fontsize=15)
    plt.xlabel('x [s]', fontsize=15)
    plt.ylabel('y [a.u.]', fontsize=15)
    plt.plot(x, y, 'b')
    plt.grid()

    plt.figure(2)
    plt.title('total fft, dft', fontsize=15)
    plt.xlabel('frequencies [Hz]', fontsize=15)
    plt.ylabel('abs(spectrum)', fontsize=15)
    plt.plot(freq_m_i, abs(dft_m_i), 'b', label='dft')
    plt.plot(freq_n_i, abs(fft_n_i), 'k', label='fft')
    plt.plot(freq_m_i, abs(fft_m_i), 'r', label='numpy')
    plt.legend(loc='best')
    plt.grid()

    plt.figure(3)
    plt.title('real fft', fontsize=15)
    plt.xlabel('frequencies [Hz]',fontsize=15)
    plt.ylabel('abs(spectrum)', fontsize=15)
    plt.plot(freq_n_r, abs(fft_n_r), 'k', label='numpy')
    plt.plot(freq_m_r, abs(fft_m_r), 'r', label='fft')
    plt.legend(loc='best')
    plt.yscale('log')
    plt.grid()

    plt.figure(4)
    plt.subplot(221)
    plt.title("abs(FFT - np.fft.fft)")
    plt.plot(abs(abs(fft_m_i)-abs(fft_n_i)), 'b')
    plt.yscale('log')
    plt.grid()

    plt.subplot(222)
    plt.title("abs(DFT - np.fft.fft)")
    plt.plot(abs(abs(dft_m_i)-abs(fft_n_i)), 'b')
    plt.yscale('log')
    plt.grid()

    plt.subplot(223)
    plt.title(r"signal - $\mathcal{F}^{-1}(\mathcal{F}(signal))$")
    plt.plot(x, abs(anti_fft-y), 'r',  label='fft')
    plt.plot(x, abs(anti_rfft-y), 'b',  label='rfft')
    plt.legend(loc='best')
    plt.yscale('log')
    plt.grid()

    plt.subplot(224)
    plt.title(r"signal - $\mathcal{F}^{-1}(\mathcal{F}(signal))$")
    plt.plot(x, abs(anti_dft-y), 'b',  label='dft')
    plt.plot(x, abs(anti_fft-y), 'r',  label='fft')
    plt.legend(loc='best')
    plt.yscale('log')
    plt.grid()

    plt.show()
