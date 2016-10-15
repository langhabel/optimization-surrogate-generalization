import numpy as np

def scale_01(x, original_range):
    """Scales arrays to the interval [0,1].

    Args:
        x: input array
        original_range: original space of the data

    Return:
        x: scaled array
    """    
    d = original_range.shape[1]
    for dim in range(d):
        zero_diff = min(original_range[0,dim], original_range[1,dim])
        x[:,dim] = (x[:,dim] - zero_diff) / abs(original_range[0,dim] - original_range[1,dim])
    return x

def scale_restore(x, original_range):
    """Scales arrays to the original space.

    Args:
        x: input array
        original_range: original space of the data

    Return:
        x: scaled array
    """    
    d = original_range.shape[1]
    for dim in range(d):
        zero_diff = min(original_range[0,dim], original_range[1,dim])
        x[:,dim] = x[:,dim] * abs(original_range[0,dim] - original_range[1,dim]) + zero_diff
    return x

def six_hump_camel(x):
    """A 2D test function.

    Args:
        x: supports single inputs of tuples/lists/arrays of shape (2,), as well as arrays of inputs of shape (n,2)
    """
    f = lambda x1,x2: (4 - 2.1*x1**2 + x1**4/3)*x1**2 + x1*x2 + (-4 + 4*x2**2)*x2**2 
    g = lambda x: f(x[0],x[1])
    if len(x) == 2:
        return g(x)
    else:
        return np.array(list(map(g, x)))

def get_grid(grid_size, dim_limits):
    """Computes equally spaced grid through feature space.

    Grows exponentially in size with the number of dimensions.

    Args:
        grid_size: the quantization per dimension
        dim_limits: for each dimension the upper and lower interval borders in shape (2, n_dims)

    Return:
        grid: equally spaced grid through feature space
    """ 
    d = np.array(dim_limits).shape[1]
    grid = np.zeros((grid_size**d, d))
    index = 0

    # TODO: Make multi-D capable.
    for i in np.linspace(dim_limits[0][0], dim_limits[1][0], grid_size):
        for j in np.linspace(dim_limits[0][1], dim_limits[1][1], grid_size):
            grid[index] = np.array([i,j])
            index += 1
    return grid
