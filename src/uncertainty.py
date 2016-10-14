"""A collection of different ways to obtain uncertainty in the sample search space.

Refer to the report for information about the general concept of uncertainty and a comparison of the respective
methods."""
import numpy as np
from sklearn.gaussian_process import GaussianProcess
from nn_regression import NN_Regression

def compute_uncertainty(train_X, train_Y, test_X, dims, method='Ensembles', nr_of_ensembles=10, training_epochs=3000):
    """This is a wrapper function around the different methods of obtaining uncertainty.

    Args:
        train_X: the training data
        train_Y: the training labels
        test_X: the test data/grid spacing
        dims: number of problem space dimensions
        method: the method used to obtain uncertainty; either Kriging, XDist, Ensembles or MCDropout
        nr_of_ensembles: the number of ensembles being created (only needed for method Ensembles)
        training_epochs: number of training epochs per ensemble/for MC Dropout (only for Ensembles and MCDropout)

    Returns:
        the uncertainty values for the samples given with test_X
    """
    if method is 'Kriging':
        uncertainty = kriging(train_X, train_Y, test_X)
    elif method is 'XDist':
        uncertainty = x_dist(train_X, test_X, dims)
    elif method is 'Ensembles':
        uncertainty = ensembles(train_X, train_Y, test_X, dims, n=nr_of_ensembles, training_epochs=training_epochs)
    elif method is 'MCDropout':
        uncertainty = mcdropout(train_X, train_Y, test_X, training_epochs=training_epochs, dropout_rate=0.9995, T=1000)
    else:
        raise ValueError('Not a supported method.')

    return uncertainty

def x_dist(train_X, test_X, dims):
    """Computes uncertainty based on the to the x axis projected distance to the next samples.

    Note: Only for 1D.
    """
    if dims != 1:
        raise ValueError('Only for 1D data.')
    holder = np.empty(len(test_X))
    for i in range(len(test_X)):
            holder[i] = np.min(np.abs(test_X[i]-train_X))**2
    return holder

#TODO test kriging in multi-D
def kriging(train_X, train_Y, test_X):
    """Computes uncertainty based on Kriging."""
    X = np.matrix(train_X).T
    y = np.matrix(train_Y).T
    TX = np.matrix(test_X).T

    gp = GaussianProcess(corr='cubic', theta0=1e-2, thetaL=1e-4, thetaU=1e-1, random_start=100)
    gp.fit(X, y)
    _, MSE = gp.predict(TX, eval_MSE=True)
    return np.sqrt(MSE)

def ensembles(train_X, train_Y, test_X, dims, n=10, training_epochs=3000):
    """Computes uncertainty based on ensembles of NN with same configuration but different weight initializations.

    Use the same network config to get the surrogate aka prediction for the strongest correlations between surrogate
    and uncertainty.
    """
    predictions = []
    for i in range(n):
        # add tweaks to default network config here
        nn = NN_Regression(data_dim=dims)
        nn.fit(train_X, train_Y, training_epochs=training_epochs)
        predictions += [nn.predict(test_X)]

    uncertainty = np.var(np.array(predictions), axis=0)
    return uncertainty

#TODO test mcdropout in multi-D
def mcdropout(train_X, train_Y, test_X, training_epochs=None, dropout_rate=None, T=None):
    """Computes uncertainty based on Gal and Ghahramani's MC Dropout."""
    nn = NN_Regression()
    nn.fit(train_X, train_Y, training_epochs=training_epochs)
    _, uncertainty= nn.predict_uncertainty(test_X, dropout_rate=dropout_rate, T=T)
    return uncertainty
