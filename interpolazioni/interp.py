"""
Code that linearly interpolates a dataset using a class
We implemente a linear and cubic(natural) interpolation
"""
import numpy as np
import matplotlib.pyplot as plt


class LinearInterp:
    '''
    1-D interpolating linear spline
    
    Parameters
    ----------
    xx : 1darray
        value on x must be strictly increasing
    yy : 1darray
        value on y
    
    Example
    -------
    >>>import numpy as np
    >>>import matplotlib.pyplot as plt
    >>>x = np.linspace(0, 1, 10)
    >>>y = np.sin(2*np.pi*x)
    >>>F = LinearInterp(x, y)
    >>>print(F(0.2))
    0.9164037243470745
    
    >>>z = np.linspace(0, 1, 100)
    >>>plt.figure(1)
    >>>plt.title('Spline interpolation')
    >>>plt.xlabel('x')
    >>>plt.ylabel('y')
    >>>plt.plot(z, F(z), 'b', label='Linear')
    >>>plt.plot(x, y, marker='.', linestyle='', c='k', label='data')
    >>>plt.legend(loc='best')
    >>>plt.grid()
    >>>plt.show()
    '''

    def __init__(self, xx, yy):
    
        self.x = xx                      # x data
        self.y = yy                      # y data
        self.N = len(xx)                 # len of data
        self.A = np.zeros(self.N-1)      # first  coefficent
        self.B = np.zeros(self.N-1)      # second coefficent
        
        if not np.all(np.diff(xx) > 0.0):
            raise ValueError('x must be strictly increasing')
        
        for i in range(self.N-1):
            # let's calculate these coefficients now 
            # so as not to always repeat the calculation
            self.A[i] = yy[i] * xx[i+1]/(xx[i+1] - xx[i])
            self.B[i] = - yy[i+1] * xx[i]/(xx[i+1] - xx[i])
            
    def __call__(self, x):
        '''
        x : float or 1darray
            when we want compute the function
        '''
        n = self.check(x)

        if n == 1 :
            
            for j in range(self.N-1):
                if self.x[j] <= x <= self.x[j+1]:
                    i = j
                    break

            A1 = - self.y[i] * x/(self.x[i+1] - self.x[i])
            B1 = self.y[i+1] * x/(self.x[i+1] - self.x[i])
            return self.A[i] + A1  + self.B[i] + B1

        else:
            F = np.zeros(n)
            for k, x1 in enumerate(x):
                
                for j in range(len(self.x)-1):
                    if self.x[j] <= x1 <= self.x[j+1]:
                        i = j
                        break

                A1 = - self.y[i] * x1/(self.x[i+1] - self.x[i])
                B1 = self.y[i+1] * x1/(self.x[i+1] - self.x[i])
                F[k] = self.A[i]+A1 + self.B[i]+B1

            return F
        
    
    def check(self, x):
        try :
            n = len(x)
            x_in = np.min(self.x) <= np.min(self.x) and np.max(self.x) >= np.max(x)
        except TypeError:
            n = 1
            x_in = np.min(self.x) <= x <= np.max(self.x)

        # if the value is not in the correct range it is impossible to count
        if not x_in :
            errore = 'Value out of range'
            raise Exception(errore)
        
        return n


class CubicSpline:
    '''
    1-D interpolating natural cubic spline
    
    Parameters
    ----------
    xx : 1darray
        value on x must be strictly increasing
    yy : 1darray
        value on y
    
    Example
    -------
    >>>import numpy as np
    >>>import matplotlib.pyplot as plt
    >>>x = np.linspace(0, 1, 10)
    >>>y = np.sin(2*np.pi*x)
    >>>F = CubicSpline(x, y)
    >>>print(F(0.2))
    0.9508316728694627
    
    >>>z = np.linspace(0, 1, 100)
    >>>plt.figure(1)
    >>>plt.title('Spline interpolation')
    >>>plt.xlabel('x')
    >>>plt.ylabel('y')
    >>>plt.plot(z, F(z), 'b', label='Cubic')
    >>>plt.plot(x, y, marker='.', linestyle='', c='k', label='data')
    >>>plt.legend(loc='best')
    >>>plt.grid()
    >>>plt.show()
    '''

    def __init__(self, xx, yy):
    
        self.x = xx                      # x data
        self.y = yy                      # y data will be the constant of polynomial
        self.N = len(xx)                 # len of data     
        alpha  = np.zeros(self.N-1)      # auxiliar array
        self.b = np.zeros(self.N-1)      # linear term
        self.c = np.zeros(self.N)        # quadratic term
        self.d = np.zeros(self.N-1)      # cubic term
        l      = np.zeros(self.N)        # auxiliar array
        z      = np.zeros(self.N)        # auxiliar array
        mu     = np.zeros(self.N)        # auxiliar array
        
        if not np.all(np.diff(xx) > 0.0):
            raise ValueError('x must be strictly increasing')
        
        dx = xx[1:] - xx[:-1]
        a = yy

        for i in range(1, self.N-1):
            alpha[i] = 3*(a[i+1] - a[i])/dx[i] - 3*(a[i] - a[i-1])/dx[i-1]
        
        l[0]  = 1.0
        z[0]  = 0.0
        mu[0] = 0.0
        
        for i in range(1, self.N-1):
            l[i]  = 2.0*(xx[i+1] - xx[i-1]) - dx[i-1]*mu[i-1]
            mu[i] = dx[i]/l[i]
            z[i]  = (alpha[i] - dx[i-1]*z[i-1])/l[i]
          
        l[self.N-1] = 1.0
        z[self.N-1] = 0.0
        self.c[self.N-1] = 0.0
        
        #Coefficient's computation
        for i in range(self.N-2, -1, -1):
            
            self.c[i] = z[i] - mu[i]*self.c[i+1]
            self.b[i] = (a[i+1] - a[i])/dx[i] - dx[i]*(self.c[i+1] + 2.0*self.c[i])/3.0
            self.d[i] = (self.c[i+1] - self.c[i])/(3.0*dx[i])
    
     
    def __call__(self, x):
        '''
        x : float or 1darray
            when we want compute the function
        '''
        n = self.check(x)

        if n == 1 :
            
            for j in range(self.N-1):
                if self.x[j] <= x <= self.x[j+1]:
                    i = j
                    break

            q = (x - self.x[i])
            return self.d[j]*q**3.0 + self.c[j]*q**2.0 + self.b[j]*q + self.y[j]
            
        else:
            F = np.zeros(n)
            for k, x1 in enumerate(x):
                
                for j in range(len(self.x)-1):
                    if self.x[j] <= x1 <= self.x[j+1]:
                        i = j
                        break

                q = (x1 - self.x[i])
                F[k] = self.d[j]*q**3.0 + self.c[j]*q**2.0 + self.b[j]*q + self.y[j]

            return F
        
    
    def check(self, x):
        try :
            n = len(x)
            x_in = np.min(self.x) <= np.min(self.x) and np.max(self.x) >= np.max(x)
        except TypeError:
            n = 1
            x_in = np.min(self.x) <= x <= np.max(self.x)

        # if the value is not in the correct range it is impossible to count
        if not x_in :
            errore = 'Value out of range'
            raise Exception(errore)
        
        return n
        
        

if __name__ == '__main__':

    x = np.linspace(0, 1,10)
    y = np.sin(2*np.pi*x)
    
    z = np.linspace(0, 1, 1000)
    
    F = LinearInterp(x, y)
    G = CubicSpline(x, y)
   
    
    print(F(0.2))
    print(G(0.2))

    
    plt.figure(1)
    plt.title('Spline interpolation')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(z, F(z), 'b', label='Linear')
    plt.plot(z, G(z), 'r', label='Cubic')
    plt.plot(x, y, marker='.', linestyle='', c='k', label='data')
    plt.legend(loc='best')
    plt.grid()
    plt.show()
