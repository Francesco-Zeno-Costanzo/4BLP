"""
Code tha implement a shallow neural network for a binary classifications.
The code is witten imposed 2 imput, 1 output e one hidden layer.
Is possible to choose the dimesions of hidden layer.
It is also possible to save plots during the run to see how the network is learning.
In this code, the learning rate is fixed
"""
import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

#=============================================================
# Loss function binary classification
#=============================================================

def Loss(Yp, Y):
    '''
    loss function, binary crosss entropy
    
    Parameters
    ----------
    Yp : 1darray
        actual prediction
    Y : 1darray
        Target
    
    Returns
    -------
    float, binary crosss entropy
    '''
    m = len(Y)
    return -np.sum(Y*np.log(Yp) + (1 - Y)*np.log(1 - Yp))/m

#=============================================================
# Activation function
#=============================================================

# Hidden layer
def g1(x):
    return np.tanh(x)
# Output layer
def g2(x):
    return 1 / (1 + np.exp(-x))

#=============================================================
# Initialization
#=============================================================

def init(n):
    '''
    Random initialization of parameters weights and biases
    
    Parameters
    ----------
    n : int
        number of neurons in the hidden layer
    
    Returns
    -------
    W1, b1 : 2darray
        weights and bias for hidden layer
    W2, b2 : 2darray
        weights and bias for output layer
    '''
    # Hidden layer
    # nx2 because 2 featurs and n neurons
    W1 = np.random.randn(n, 2)
    b1 = np.random.rand(n, 1)
    # Output layer
    # 1xn because 1 output and n neurons
    W2 = np.random.randn(1, n)
    b2 = np.random.rand(1, 1)
    return W1, b1, W2, b2

#=============================================================
# Network prediction function
#=============================================================

def predict(X, W1, b1, W2, b2):
    """
    Function that returns the prediction of the network
    
    Parameters
    ----------
    X : 2darray
        data, featurs
    W1, b1, W2, b2 : 2darray
        parameter of the network
    
    Returns
    -------
    A1 : 1d array
        intermediate prediction
    A2 : 1d array
        final prediction
    """
    # Hidden layer
    Z1 = W1 @ X + b1
    A1 = g1(Z1)
    # Output layer
    Z2 = W2 @ A1 + b2
    A2 = g2(Z2)
    
    return A1, A2

#=============================================================
# Backpropagation function
#=============================================================

def backpropagation(X, Y, step, A1, A2, W1, b1, W2, b2):
    '''
    Backpropagation function.
    Update weights and biases with gradient descendent
    all the quantities came from taking the derivative of the Loss
    
    Y : 1darray
        Target
    step : float
        learning rate
    A1, A2 : 1darray
        predictions of the network
    W1, b1, W2, b2 : 2darray
        parameter of the network
    '''
    m = len(Y)
    # Output layer 
    dLdZ2 = (A2 - Y)
    dLdW2 = dLdZ2 @ A1.T / m
    dLdb2 = np.sum(dLdZ2, axis=1)[:, None] / m
    # Hidden layer
    dLdZ1 = W2.T @ dLdZ2 * (1 - A1**2)
    dLdW1 = dLdZ1 @ X.T / m
    dLdb1 = np.sum(dLdZ1, axis=1)[:, None] / m
    
    # Update of parameters
    W1 -= step * dLdW1
    b1 -= step * dLdb1
    W2 -= step * dLdW2
    b2 -= step * dLdb2
    
    return W1, b1, W2, b2

#=============================================================
# Accuracy mesuraments
#=============================================================

def accuracy(Yp, Y):
    '''
    accuracy of prediction. We use:
    accuracy = 1 - | sum ( prediction - target )/target_size |
    
    Parameters
    ----------
    Yp : 1darray
        actual prediction
    Y : 1darray
        Target
    
    Returns
    -------
    a : float
        accuracy
    '''
    m = len(Y)
    a = 1 - abs(np.sum(Yp.ravel() - Y)/m)
    return a
    
#=============================================================
# Train of the network
#=============================================================

def train(X, Y, n_epoch, neuro, step, sp=False, verbose=True):
    '''
    function for the training of the network
    
    Parameters
    ----------
    X : 2darray
        data, featurs
    Y : 1darray
        Target
    n_epoch : int
        number of epoch
    neuro : int
        number of neurons in the hidden layer
    step : float
        learning rate
    sp : boolean, optional, default False
        if True a plot of boundary is saved each 100 epoch
        usefull for animations 
    verbose : boolean, optional, default True
        if True print loss and accuracy each 100 epoch
    
    Returns
    -------
    result : dict
        params -> W1, b1, W2, b2 weights and bias of network
        train_Loss -> loss on train data
        valid_Loss -> loss on validation data
    '''

    W1, b1, W2, b2 = init(neuro)
    L_t = np.zeros(n_epoch) # training loss
    L_v = np.zeros(n_epoch) # validation loss
    N = X.shape[1]          # total number of data
    M = N//4                # nuber of data for validation
     
    # split dataset in validation and train 
    X_train, Y_train = X[:, :N-M ], Y[:N-M ] 
    X_valid, Y_valid = X[:,  N-M:], Y[ N-M:]
    
    for i in range(n_epoch):
        # train
        A1, A2 = predict(X_train, W1, b1, W2, b2)
        L_t[i] = Loss(A2, Y_train)
        # validation
        _, Yp = predict(X_valid, W1, b1, W2, b2)
        L_v[i] = Loss(Yp, Y_valid)
        # update
        W1, b1, W2, b2 = backpropagation(X_train, Y_train, step, A1, A2, W1, b1, W2, b2)
              
        if not i % 100:
            if sp : plot(X_train, Y_train, (W1, b1, W2, b2), i)
            
            if verbose:
                acc = accuracy(A2, Y_train)
                print(f'Loss = {L_t[i]:.5f}, accuracy = {acc:.5f}, epoch = {i} \r', end='')
            
    if verbose: print()
    
    result = {'params'     : (W1, b1, W2, b2),
              'train_Loss' : L_t,
              'valid_Loss' : L_v,
             }
        
    return result

#=============================================================
# Plot
#=============================================================

def plot(X, Y, par, k, close=True, i=0):
    '''
    Plot of boundary
    
    Parameters
    ----------
    X : 2darray
        data, featurs
    Y : 1darray
        Target
    par : tuple
        parameter of the network (W1, b1, W2, b2)
    close : boolean, optional, defautl True
        if we want to see only the last epoch we don't
        want the figure to be closed, while we have to
        make many for the animation the various figures
        must be closed
    i : int
        index of figure
    '''
    
    # bound of plot
    x_min, x_max = np.min(X[0, :]), np.max(X[0, :])
    y_min, y_max = np.min(X[1, :]), np.max(X[1, :])
    
    # data to predict to create the decision boundary
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = predict(np.array([xx.ravel(), yy.ravel()]), *par)[-1]
    Z = Z.reshape(xx.shape)
    
    # Plot boundary as a contour plot
    fig = plt.figure(i, figsize=(6, 6))
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.contourf(xx, yy, Z, cmap='plasma')
    plt.scatter(X[0, :], X[1, :], c=Y, cmap='plasma', s=8)
    plt.title(f'Neural network learning epoch={k}', fontsize=15)
    plt.ylabel('x2', fontsize=15)
    plt.xlabel('x1', fontsize=15)
    if close : plt.savefig(f'{k}.pdf')
    if close : plt.close(fig)

#=============================================================
# GIF
#=============================================================

def GIF(path, name, ext='.png'):
    '''
    function to create a gif from several plot
    
    Parameters
    ----------
    path : string
        path of the varius plot
    name : string
        name of the gif
    ext : string, optional, default .png
        type of file
    '''
    
    fig_path = path + '/*' + ext
    gif_path = path + '/' + name + '.gif'
    
    frames=[]
    imgs = sorted(glob.glob(fig_path))
    imgs.sort(key=len)
    for i in imgs:
        new_frame = Image.open(i)
        frames.append(new_frame)

    frames[0].save(gif_path, format='GIF', \
    append_images=frames[:],save_all=True,duration=100,loop=0)



if __name__ == '__main__':
    
    np.random.seed(69420)

#=============================================================
#   Creation of dataset 
#=============================================================

    N = 5000                          # Total number of points
    M = 1000                          # number of test and validation points
    X = np.random.random(size=(2, N)) # Two features
    Y = np.ones(N)                    # one Target
    
    # Selection of some regions where the target value is different
    for x1, x2, i in zip(X[0, :], X[1, :], range(N)):
        if np.sqrt( (x1 - 0.3)**2 + (x2 - 0.3)**2 ) < 0.2:
            Y[i] = 0
        if np.sqrt( (x1 - 0.65)**2 + (x2 - 0.7)**2 ) < 0.2:
            Y[i] = 0
    
    # split dataset in test and train 
    # Part of train data will be used for validation
    X_train, Y_train = X[:, :N-M ], Y[:N-M ] 
    X_test,  Y_test  = X[:,  N-M:], Y[ N-M:]
    
#=============================================================
#   Parameter of computation
#=============================================================

    n_epoch = 6000 + 1  # number of epoch
    n_neuro = 20        # number of neurons for the hidden layer
    lr_rate = 1.5         # learning rate
    sp_gif  = 0         # save plot and make gif

#=============================================================
#   Train of the network
#=============================================================

    result = train(X_train, Y_train, n_epoch, n_neuro, lr_rate, sp_gif)
    
    parameters = result['params']
    L_t = result['train_Loss']
    L_v = result['valid_Loss']
    
    if sp_gif : GIF('frames', 'NN')
    print(f'Loss on train      set = {L_t[-1]:.5f}')
    print(f'Loos on validation set = {L_v[-1]:.5f}')

#=============================================================
#   Test of the network
#=============================================================    
    _, A2 = predict(X_test, *parameters)
    acc  = accuracy(A2, Y_test)
    loss = Loss(A2, Y_test)
    plot(X_test, Y_test, parameters, k=n_epoch, close=True, i=1)
    
    print(f'Loss     on test  set = {loss:.5f}')
    print(f"Accuracy on test  set = {acc:.5f}")
    
    plt.figure(2)
    plt.plot(np.linspace(1, n_epoch, n_epoch), L_t, 'b', label='train Loss')
    plt.plot(np.linspace(1, n_epoch, n_epoch), L_v, 'r', label='validation loss')
    plt.title('Binary cross entropy', fontsize=15)
    plt.xlabel('epoch',  fontsize=15)
    plt.ylabel('Loss',  fontsize=15)
    plt.legend(loc='best')
    plt.grid()
    plt.yscale('log')
    plt.savefig("Loss1.pdf")
    plt.show()
     
    
