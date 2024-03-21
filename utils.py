import numpy as np
import matplotlib.pyplot as plt

def compute_crossing_percentage(quantiles):
    """
    Compute the crossing percentage.

    Arguments:
        quantiles: An array where each row contains the quantiles for a single
            data point. The quantiles are assumed to be in decreasing order.
    """
    crossings = 0
    max_crossings = (len(quantiles[0]) -1) * len(quantiles)
    for loc in quantiles:
        for i in range(1, len(loc)):
            crossings += (loc[i] > loc[i-1])
    return crossings / max_crossings * 100



def scoring_loss(estimator, X, y, taus):
    predicted_quantiles = estimator.predict(X)
    assert np.shape(predicted_quantiles[:, 0]) == np.shape(y[:, 0])
    total = 0
    for i, tau in enumerate(taus):
        y_values = y[:, 0]
        q_values = predicted_quantiles[:, i]
        total += pinball_loss(y_values, q_values, tau)
    return -total

def pinball_loss_total(quantiles, Y, taus):
    assert np.shape(quantiles[:, 0]) == np.shape(Y)
    total = 0
    for i, tau in enumerate(taus):
        q_values = quantiles[:, i]
        total += pinball_loss(Y, q_values, tau)
    return total / len(taus)

def pinball_loss(y, q, tau):
    assert np.shape(y) == np.shape(q)
    u_values = y - q
    total = 0
    for u in u_values:
        if u > 0:
            total += tau * u
        if u < 0:
            total += (tau - 1) * u
    return total


def load_data(directory):
    """
    Load data from given directory

    This function was adapted from a similar code snippet made by
    Yarin Gal.
    """
    _DATA_FILE = "./UCI_Datasets/" + directory + "/data/data.txt"
    _INDEX_FEATURES_FILE = "./UCI_Datasets/" + directory + "/data/index_features.txt"
    _INDEX_TARGET_FILE = "./UCI_Datasets/" + directory + "/data/index_target.txt"
    index_features = np.loadtxt(_INDEX_FEATURES_FILE)
    index_target = np.loadtxt(_INDEX_TARGET_FILE)
    data = np.loadtxt(_DATA_FILE)
    X = data[:, [int(i) for i in index_features.tolist()]]
    Y = data[:, int(index_target.tolist())]
    return X, Y

def plot_quantiles(quantiles, Y, taus, title=None, save_loc=None):
    y_values = np.zeros(len(taus))
    for i in range(len(taus)):
        assert np.shape(Y) == np.shape(quantiles[:, i])
        y_values[i] = np.mean(Y < quantiles[:, i])
    plt.figure(dpi=400, figsize=(8, 6))
    if title:
        plt.title(title)

    plt.plot(taus, y_values, marker='o')
    plt.plot([0, 1], [0, 1], linestyle=':', color='black')
    plt.gca().set_aspect('equal', 'box')
    plt.ylabel(r'$P(Y < \hat{q})$')
    plt.xlabel(r'$\hat{q}$')
    plt.xlim((0, 1))
    plt.ylim((0, 1))

    if save_loc:
        plt.savefig(save_loc)
    else:
        plt.show()
    plt.close()

def convert_to_quantiles(outputs):
    n_dim = len(outputs[0])
    f = 4 / (n_dim) * np.exp(2)
    predictions = np.zeros_like(outputs)
    predictions[:, 0] = outputs[:, 0]
    for i in range(1, n_dim):
        predictions[:, i] = predictions[:, i-1] + f * np.exp(outputs[:, i])
    return predictions

def convert_to_quantiles_2(outputs, s2=1, f=1):
    n_dim = len(outputs[0])
    # f = 4 / (n_dim) * np.exp(2)
    # f = 1
    predictions = np.zeros_like(outputs)
    predictions[:, 0] = outputs[:, 0]
    for i in range(1, n_dim):
        predictions[:, i] = predictions[:, i-1] + f * np.log(1+np.exp(outputs[:, i] / s2))
    return predictions

def coverage(lower_bounds, upper_bounds, Y):
    uppercorrect = Y < upper_bounds
    lowercorrect = Y > lower_bounds
    return np.round(100 * np.mean(uppercorrect * lowercorrect), 2)