"""Plotting function for the example 2D problem.
"""
import numpy as np
import math
import utils
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


def plot(proof, slice_, grid_size, dim_limits, slice_samples_X, slice_samples_Y, past_slices, samples_X, samples_Y):
    """Plots the state after a sample evaluation round.

    Figure is 2 rows and 2 columns
        - First row shows 2D plots:
          True function and surrogate representation with current slice.
        - Second row shows 1D plots:
          True function + Expected Improvement and
          True function + Surrogate + Uncertainty 
    Surrogate representation print samples as well.

    Title contains 2D as well as 1D slice test errors.
    """
    prediction, uncertainty, expected_improvement = proof[0], proof[1], proof[2]
    
    fig = plt.figure(figsize=(20, 20))
    gs = gridspec.GridSpec(2, 2)
    
    d = np.array(dim_limits).shape[1]
    grid = utils.get_grid(grid_size, dim_limits)
    Y = utils.six_hump_camel(grid)
    grid = utils.scale_01(grid, dim_limits)
    
    indices = np.isclose(grid[:, d-1], slice_)
    grid_1D = np.linspace(0, 1, grid_size)
    true_1D = Y[indices]
    pred_1D = prediction[indices]
    uncert_1D = uncertainty[indices]
    expimp_1D = expected_improvement[indices]

    xlim = (-0.01, 1.01)

    # True function
    ax = fig.add_subplot(gs[0], projection='3d')
    ax.plot(grid_1D, np.ones(grid_size)*slice_, true_1D, 'b-', label='Current Slice', lw=2)
    for i in past_slices:
        indices_ = np.isclose(grid[:, d-1], i)
        true_1D_ = Y[indices_]
        ax.plot(grid_1D, np.ones(grid_size)*i, true_1D_, 'b--', alpha=1.0-(i/1.6), lw=2)
    _ax_plot3D(ax, grid, Y, cm.Blues)
    ax.scatter(samples_X[:, 0], samples_X[:, 1], samples_Y, c='r', label='Samples', s=50)
    ax.set_title('Original Function')
    ax.legend(loc='upper left')

    # Surrogate + slice
    ax = fig.add_subplot(gs[1], projection='3d')
    ax.plot(grid_1D, np.ones(grid_size)*slice_, pred_1D, 'b-', label='Current Slice', lw=2)
    for i in past_slices:
        indices_ = np.isclose(grid[:, d-1], i)
        pred_1D_ = prediction[indices_]
        ax.plot(grid_1D, np.ones(grid_size)*i, pred_1D_, 'b--', alpha=1.0-(i/1.6), lw=2)
    _ax_plot3D(ax, grid, prediction, cm.Greens)
    ax.scatter(slice_samples_X[:, 0], slice_samples_X[:, 1], slice_samples_Y, c='r', label='Samples in Slice', s=50)
    ax.set_title('Surrogate Model')
    ax.legend(loc='upper left')
    ax.legend(loc='lower right')

    # True function + Expected Improvement
    ax = fig.add_subplot(gs[2])
    ax.plot(grid_1D, true_1D, 'r--', label='Original Curve')
    ax.plot(grid_1D, expimp_1D/np.max(expimp_1D), '-', color='darkred', label='Expected Improvement')
    ax.set_xlim(xlim)
    ax.legend(loc='upper left')

    # True function + Surrogate + Uncertainty 
    ax = fig.add_subplot(gs[3])
    ax.plot(grid_1D, true_1D, 'r--', label='Original Curve')
    ax.plot(grid_1D, pred_1D, 'b-', label='Surrogate Model')
    ax.plot(grid_1D, uncert_1D/np.max(uncert_1D), '-', color='orange', label='Uncertainty')
    ax.plot(slice_samples_X[:, 0], slice_samples_Y, 'ko', label='Samples')
    ax.set_xlim(xlim)
    ax.legend(loc='upper left')

    plt.show()


def _ax_plot3D(ax, X, Y, cmap):
    n = int(math.sqrt(X.shape[0]))
    ax.plot_surface(X[:, 0].reshape(n, n), X[:, 1].reshape(n, n), Y.reshape(n, n),
                    cmap=cmap, alpha=0.65, rstride=2, cstride=2, linewidth=0.01, antialiased=True)
    ax.set_xlabel('x1', fontsize=15)
    ax.set_ylabel('x2', fontsize=15)
    ax.set_zlabel('y', fontsize=15)
