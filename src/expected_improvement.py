import numpy as np
from scipy.stats import norm


def compute_expected_improvement_forrester(prediction, uncertainty, y_min, grid):
    """Computes the expected improvement in every grid point.

    According to formula (81), Section 4.3.1, Recent advances in surrogate-based optimization, Forrester and Keane.

    Args:
        prediction: the predicted fit to the function based upon the same samples, given in the space of 'grid'
        uncertainty: the respective uncertainty at all gridpoints based upon the samples
        y_min: the minimum function value of the samples in the slice
        grid: the grid used to represent the space

    Return:
        expected_improvement: the expected improvement in every grid point
    """

    # switch to nomenclature of paper
    x = grid
    y_hat = prediction
    s = uncertainty

    # if s==0 expected improvement is 0
    binary = np.sign(s)

    # calibrate "s"
    # TODO: automate (it is hardcoded now to suite the demo example)
    s /= 2.0

    # compute expected improvement
    dif = y_min - y_hat
    arg = np.divide(dif, s)
    cdf = norm.cdf(arg)
    pdf = norm.pdf(arg)
    E = binary * (dif * cdf + s * pdf)

    return E
