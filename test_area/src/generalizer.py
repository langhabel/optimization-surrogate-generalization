"""Main program which optimizes n dimensional search spaces. The Generalizer optimizes faster by maintaining a n+1 dimensional surrogate model which incorporates the samples of already solved similar problems with shared features. The additional axis is not vital to solve each optimization problem but connects similar problems from the same class. We thus speak from generalizing between optimization problems.

We call the n dimensional problems (hyper)slices of the n+1 dimensional surrogate space (intuition: 1D slices of a 2D problem space).

The algorithm makes use of three exchangeable modules as defined in the function defs:
_get_expected_improvement
    The function responsible for attributing each grid point with the fitness of improving the optima, if elected as next sampling point.
_get_generalization_surrogate
    The surrogate model used to envelop the related optimization problems.
_get_uncertainty
    Computes the uncertainty inherent in the surrogate model. The technique used should depend on the used surrogate model.
"""
import numpy as np
from uncertainty import compute_uncertainty
#from expected_improvement import compute_expected_improvement_forrester
from surrogate_model import NN_Regression
import utils


class Generalizer:
    """Module to optimize a 1D function with continuous 2D slicing. TODO: extend to arbitrary dimensions.

    Note: All coordinates are internally scaled to the range 0 to 1.

    Attributes:
        nn: the network serving as generalization surrogate model
        samples_X: list of all samples we have, that support the generalization surrogate, in (n+1)-D coordinates
        samples_Y: list of the respective labels
        d: dimension of the data (n+1)
        grid: the discretized search space of the surrogate
        generalization_surrogate
    """

    def __init__(self,
                 dim_limits,
                 initial_samples_X=None,
                 initial_samples_Y=None,
                 grid_size=100):
        """
        Args:
            dim_limits: list of shape (2, n_dims) that gives for each of the n+1 dimension an upper and lower limit 
            initial_samples_X: list of all samples we have, that support the generalization surrogate, in (n+1)-D coordinates
            initial_samples_Y: list of the respective labels
            grid_size: the quantization per dimension
        """
        self.d = np.array(dim_limits).shape[1]

        self.grid = utils.get_grid(grid_size, dim_limits)
        self.grid = utils.scale_01(self.grid, dim_limits)
        
        self.samples_X = None
        self.samples_Y = None
        
        # init surrogate
        self.generalization_surrogate = self._get_generalization_surrogate()
        if initial_samples_X is not None and initial_samples_Y is not None:
            self.samples_X = np.array(initial_samples_X)
            self.samples_Y = np.array(initial_samples_Y)

    def incorporate_new_sample(self, x_new, y_new):
        """Used to incorporate new samples in the surrogate model during the optimization process.

        Args:
            x_new: a new sample location given in scaled n-D coordinates
            y_new: the respective label
        """
        # add sample to database
        if self.samples_X is None:
            self.samples_X = np.matrix(x_new)
            self.samples_Y = np.array([y_new])
        else:
            self.samples_X = np.vstack((self.samples_X, np.matrix(x_new)))
            self.samples_Y = np.append(self.samples_Y, np.array(y_new))
        
    def get_next_sampling_location(self, slice_):
        """Computes the best location to evaluate the next sample within the given slice. 
        
        Args:
            slice_: the (n+1)-th dimension value   
            
        Return:
            scaled_coordinates: the ideal next sampling location in scaled coordinates
            proof: tuple of (prediction, uncertainty, expected_improvement) backing the result
        """
        # fit surrogate model to latest database
        self.generalization_surrogate.fit(self.samples_X, self.samples_Y, verbose=False)
        
        # evaluate (n+1)-D prediction
        prediction = self.generalization_surrogate.predict(self.grid)
            
        # evaluate (n+1)-D uncertainty
        uncertainty = self._get_uncertainty()
            
        # evaluate (n+1)-D expected improvement
        _, y = self.get_samples_for(slice_)
        if len(y) != 0:
            y_min = np.min(y)
        else:
            y_min = 0.0
        e_i = self._get_expected_improvement(prediction, uncertainty, y_min)
        
        proof = (prediction, uncertainty, e_i)
        
        # get maximum index within n-D slice
        e_i[~np.isclose(self.grid[:, self.d-1], slice_)] = 0
        if np.allclose(e_i, 0):
            prediction_ = np.copy(prediction)
            prediction_[~np.isclose(self.grid[:, self.d-1], slice_)] = float('inf')
            new_sample_index = np.argmin(prediction_)
        else:
            new_sample_index = np.argmax(e_i)
            
        scaled_coordinates = self.grid[new_sample_index]
        return scaled_coordinates, proof
    
    def get_samples_for(self, slice_):
        """Searches and returns all samples in the dataset that lie in a given slice.
        
        Args:
            slice_: the (n+1)-th dimension value  
            
        Return:
            slice_samples_X, slice_samples_Y: the samples and respective labels belonging to the given slice
        """
        indices = np.isclose(self.samples_X[:, self.d-1], slice_)
        indices = np.asarray(indices).squeeze()
        return self.samples_X[indices], self.samples_Y[indices]

    def _get_expected_improvement(self, prediction, uncertainty, y_min):
        """Computes the expected improvement in every grid point.

        According to formula (81), Section 4.3.1, Recent advances in surrogate-based optimization, Forrester and Keane.

        Args:
            prediction: the predicted fit to the function based upon the same samples, given in the space of 'grid'
            uncertainty: the respective uncertainty at all gridpoints based upon the samples
            y_min: the minimum function value of the samples in the slice

        Return:
            expected_improvement: the expected improvement in every grid point
        """
        #E = compute_expected_improvement_forrester(prediction, uncertainty, y_min, self.grid)

	'''
	The expected improvement part is not yet fully implemented. It is rather hard coded.
        '''
	from scipy.stats import norm
        # switch to nomenclature of paper
        x = self.grid
        y_min = np.min(self.samples_Y)
        y_hat = prediction
        s = compute_uncertainty(self.samples_X, self.samples_Y, self.grid, self.d, method='Ensembles',
                                          nr_of_ensembles=10, training_epochs=1000)
	#compute_uncertainty(self.samples_X, self.samples_Y, self.grid, method=self.method, ensembles_n = self.ensembles_n, 			training_epochs_ensembles=self.training_epochs_ensembles, twod=self.twod)
        # tests if s is zero, case distinction
        binary = np.sign(s)
        if not np.all(np.greater_equal(binary,0)):
            raise Error('Uncertainty was negative.')

        s = s/2.0
            
        # formula
        dif = y_min - y_hat
        arg = np.divide(dif, s)
        cdf = norm.cdf(arg)
        pdf = norm.pdf(arg)
        E = binary * (dif*cdf + s*pdf)

        return E

    def _get_generalization_surrogate(self):
        """The surrogate model used to envelop the related optimization problems.

        We use a neural network for nonlinear regression.

        Return:
            nn: a nonlinear neural network regressor
        """
        nn = NN_Regression(data_dim=self.d)
        return nn

    def _get_uncertainty(self):
        """Computes the uncertainty globally using ensembles of neural networks.

        Return:
            uncertainty: variance of the ensemble predictions
        """

        uncertainty = compute_uncertainty(self.samples_X, self.samples_Y, self.grid, self.d, method='Ensembles',
                                          nr_of_ensembles=10, training_epochs=1000)
        return uncertainty
