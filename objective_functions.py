import numpy as np

pi = np.pi


def arctan_loss(y_true, y_pred, taus, s=0.1):
    """Compute the arctan pinball loss.

    Note that XGBoost outputs the predictions in a slightly peculiar manner.
    Suppose we have 100 data points and we predict 10 quantiles. The predictions
    will be an array of size (1000 x 1). We first resize this to a (100x10) array
    where each row corresponds to the 10 predicted quantile for a single data
    point. We then use a for-loop (over the 10 columns) to calculate the gradients
    and second derivatives. Legibility was chosen over efficiency. This part
    can be made more efficient.

    Parameters:
        y_true: An array containing the true observations.
        y_pred: An array containing the predicted quantiles
        taus: A list containing the true desired coverage of the quantiles
        s: A smoothing parameter

    Returns:
        grad: An array containing the (negative) gradients with respect to y_pred.
        hess: An array containing the second derivative with respect to y_pred.
    """
    size = len(y_true)
    n_dim = len(taus)  # The number of columns
    n_rows = size // n_dim

    # Resize the predictions and targets.
    # Each column corresponds to a quantile, each row to a data point.
    y_pred = np.reshape(y_pred, (n_rows, n_dim))
    y_true = np.reshape(y_true, (n_rows, n_dim))

    # Calculate the differences
    u = y_true - y_pred

    # Calculate the gradient and second derivatives
    grad = np.zeros_like(y_pred)
    hess = np.zeros_like(y_pred)
    z = u / s
    for i, tau in enumerate(taus):
        x = (1 + z[:, i]**2)
        grad[:, i] = tau - 0.5 + 1 / pi * np.arctan(z[:, i]) \
                     + z[:, i] / (pi) * x ** -1
        hess[:, i] = (2 / (pi * s) * x ** (-2))

    # Reshape back to the original shape.
    grad = grad.reshape(size)
    hess = hess.reshape(size)
    return -grad / n_dim, hess / n_dim


def smooth_pinball_loss(y_true, y_pred, taus, s=0.05):
    """Compute the smooth pinball loss.

    The idea is the same as for the arctan pinball loss but with a
    different smooth approximation. However, the second derivative becomes
    far too small, making it a bad candidate for xgboost. I would not
    recommend using this objective function.

    Parameters:
        y_true: An array containing the true observations.
        y_pred: An array containing the predicted quantiles
        taus: A list containing the true desired coverage of the quantiles
        s: A smoothing parameter

    Returns:
        grad: An array containing the (negative) gradients with respect to y_pred.
        hess: An array containing the second derivative with respect to y_pred.
    """
    size = len(y_true)
    n_dim = len(taus)
    n_rows = size // n_dim
    y_pred = np.reshape(y_pred, (n_rows, n_dim))
    y_true = np.reshape(y_true, (n_rows, n_dim))
    u = y_true - y_pred
    grad = np.zeros_like(y_pred)
    hess = np.zeros_like(y_pred)
    for i, tau in enumerate(taus):
        grad[:, i] = -tau + (1+np.exp(u[:, i]) ** (1 / s))**(-1)
        hess[:, i] = (np.exp(-0.5 * u[:, i]) ** (1 / s) + np.exp(0.5 * u[:, i]) **(1/s)) ** (-2) * 1 / s
    grad = grad.reshape(size)
    hess = hess.reshape(size)
    return grad, hess







