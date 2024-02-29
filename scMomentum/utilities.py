# utilities.py

import numpy as np
from scipy.interpolate import griddata
from scipy.optimize import minimize
from scipy.special import hyp2f1 as hyper
from scipy.signal import convolve2d


def sigmoid(x, s, n):
    """
    Compute the sigmoid function for given input x, threshold s, and exponent n.

    Args:
        x (np.ndarray): Input array for which to compute the sigmoid function.
        s (float or np.ndarray): Threshold parameter of the sigmoid. The point at which the sigmoid
                                 transitions from its minimum value to its maximum value.
        n (float): Exponent parameter controlling the steepness of the sigmoid curve.

    Returns:
        np.ndarray: The sigmoid function applied to each element of x.
    """
    # Ensure inputs are numpy arrays for element-wise operations
    x = np.asarray(x)
    s = np.asarray(s)
    
    # Compute the sigmoid function
    return x**n / (x**n + s**n)

def fit_k(g):
    """
    Fit a threshold parameter k for a heavyside function based on data g by aligning
    it with the most rapid increase in the empirical cumulative distribution function (ECDF) of g.

    Args:
        g (np.ndarray): Input data array from which to compute the threshold k.

    Returns:
        float: Optimized threshold parameter k.
    """
    # Ensure g is a numpy array and remove any NaN values
    g = np.asarray(g)
    g = g[~np.isnan(g)]
    
    # Sort g to compute the ECDF
    sorted_g = np.sort(g)
    
    # Compute ECDF - each point in sorted_g corresponds to a step in the ECDF
    ecdf = np.arange(1, len(sorted_g) + 1) / len(sorted_g)
    
    # Compute the gradient of the ECDF to find the most rapid increase
    gradient_ecdf = np.gradient(ecdf, sorted_g)
    
    # Find the index of the maximum gradient, which corresponds to the optimal k
    max_gradient_index = np.argmax(gradient_ecdf)
    
    # The optimal k is the value in sorted_g at the index of the maximum gradient
    k = sorted_g[max_gradient_index]
    
    return k

def fit_sigmoid(g, min_th=0.05):
    """
    Fit a sigmoid curve to the given data using a fast fitting method with a fallback to a more accurate method if needed.
    """
    length = len(g)
    min_th *= max(g)
    offset = np.sum(g < min_th) / length
    valid_data = g[g > min_th]

    # Fast fit
    x = np.sort(valid_data)
    y = np.linspace(0, 1, len(valid_data))
    tx = np.log(x)
    ty = np.log(y / (1 - y))

    valid = np.isfinite(tx) & np.isfinite(ty)
    tx, ty = tx[valid], ty[valid]

    A = np.vstack([tx, np.ones(len(tx))]).T
    n, b = np.linalg.lstsq(A, ty, rcond=None)[0]
    m1 = np.exp(-b / n)
    mse = np.mean((sigmoid(x, m1, n) - y) ** 2)

    return m1, n, offset, mse


def int_sig_act_inv(x, s, n, verbose=False):
    """
    Compute the integral of the inverse sigmoid activation function.

    Args:
        x (np.ndarray): Input data array.
        s (float): Sigmoid threshold parameter.
        n (float): Sigmoid steepness parameter.
        verbose (bool): If True, prints intermediate computation results.

    Returns:
        np.ndarray: Integral of the inverse sigmoid activation.
    """
    z = -(n / (n - 1)) * s * hyper(-1 / n, (n - 1) / n, (2 * n - 1) / n, 1)
    z = z[None, :]
    n = n[None, :]
    s = s[None, :]
    z1 = -n * s * (1 - x) ** ((n - 1) / n) * hyper(-1 / n, (n - 1) / n, (2 * n - 1) / n, 1 - x) / (n - 1)
    
    if verbose:
        print(z[0])
        print(z1)
    
    return z1 - z

def d_sigmoid(x, s, n):
    """
    Compute the derivative of the sigmoid function with respect to x.

    Args:
        x (np.ndarray): Input array for which to compute the derivative.
        s (float): Threshold parameter of the sigmoid.
        n (float): Exponent parameter controlling the steepness of the sigmoid curve.

    Returns:
        np.ndarray: The derivative of the sigmoid function applied to each element of x.
    """
    sig = sigmoid(x, s, n)
    return np.nan_to_num(sig * (1 - sig) / x)

def soften(z, n_filt=5):
    """
    Apply a softening filter to an array z, with a specified filter size n_filt.
    The filter applied is a uniform filter, except for the center, which is set to 0.

    Args:
        z (np.ndarray): Input 2D array to be softened.
        n_filt (int): Size of the square filter, must be an odd number to have a center.

    Returns:
        np.ndarray: The softened 2D array.
    """
    # Create a uniform filter of size n_filt x n_filt
    filt = np.ones((n_filt, n_filt)) / (n_filt**2 - 1)
    
    # Set the center of the filter to 0
    filt[n_filt // 2, n_filt // 2] = 0
    
    # Apply the filter to the input array z using 2D convolution
    softened_z = convolve2d(z, filt, mode='same')
    
    return softened_z

def rezet(gridX, gridY, energySurface, points):
    """
    Interpolate energy values from a grid onto specific points.

    Args:
        gridX (np.ndarray): The grid's X coordinates.
        gridY (np.ndarray): The grid's Y coordinates.
        energySurface (np.ndarray): The energy values at each point on the grid.
        points (np.ndarray): The points at which to interpolate energy values, shape (N, 2).

    Returns:
        np.ndarray: The interpolated energy values at the given points.
    """
    points = np.array(points)
    grid_points = np.array([gridX.ravel(), gridY.ravel()]).T
    energy_values = griddata(grid_points, energySurface.ravel(), points, method='cubic')
    return energy_values
