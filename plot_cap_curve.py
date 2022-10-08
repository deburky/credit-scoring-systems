"""
Module with plots for visualizing model performance.
"""

# Gabriel S. Gonçalves <gabrielgoncalvesbr@gmail.com>
# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2020

# Edits by Denis Burakov (https://github.com/deburky)

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.utils import check_array
from sklearn.utils import check_consistent_length

def _check_arrays(y, y_pred):
    y = check_array(y, ensure_2d=False, force_all_finite=True)
    y_pred = check_array(y_pred, ensure_2d=False, force_all_finite=True)

    check_consistent_length(y, y_pred)

    return y, y_pred


def _check_parameters(title, xlabel, ylabel, savefig, fname):
    if title is not None and not isinstance(title, str):
        raise TypeError("title must be a string or None; got {}."
                        .format(title))

    if xlabel is not None and not isinstance(xlabel, str):
        raise TypeError("xlabel must be a string or None; got {}."
                        .format(xlabel))

    if ylabel is not None and not isinstance(ylabel, str):
        raise TypeError("ylabel must be a string or None; got {}."
                        .format(ylabel))

    if not isinstance(savefig, bool):
        raise TypeError("savefig must be a boolean; got {}.".format(savefig))

    if fname is not None and not isinstance(fname, str):
        raise TypeError("fname must be a string or None; got {}."
                        .format(fname))

    if savefig is True and fname is None:
        raise ValueError("fname must be provided if savefig is True.")

# changes in model_name and line_color to plot multiple curves

def plot_cap(y, y_pred, model_name='Model', title=None, xlabel=None, ylabel=None,
             savefig=False, line_color='g', fname=None, **kwargs):
    """Plot Cumulative Accuracy Profile (CAP).
    Parameters
    ----------
    y : array-like, shape = (n_samples,)
        Array with the target labels.
    y_pred : array-like, shape = (n_samples,)
        Array with predicted probabilities.
    title : str or None, optional (default=None)
        Title for the plot.
    xlabel : str or None, optional (default=None)
        Label for the x-axis.
    ylabel : str or None, optional (default=None)
        Label for the y-axis.
    savefig : bool (default=False)
        Whether to save the figure.
    fname : str or None, optional (default=None)
        Name for the figure file.
    **kwargs : keyword arguments
        Keyword arguments for matplotlib.pyplot.savefig().
    """
    y, y_pred = _check_arrays(y, y_pred)

    _check_parameters(title, xlabel, ylabel, savefig, fname)

    n_samples = y.shape[0]
    n_event = np.sum(y)

    idx = y_pred.argsort()[::-1][:n_samples]
    yy = y[idx]

    p_event = np.append([0], np.cumsum(yy)) / n_event
    p_population = np.arange(0, n_samples + 1) / n_samples

    auroc = roc_auc_score(y, y_pred)
    gini = auroc * 2 - 1

    # Define the plot settings
    if title is None:
        title = "Cumulative Accuracy Profile (CAP)"
    if xlabel is None:
        xlabel = "Cumulative share of observations"
    if ylabel is None:
        ylabel = "Cumulative share defaults"

    plt.plot([0, 1], [0, 1], color='red', linewidth=2)
    plt.plot([0, n_event / n_samples, 1], [0, 1, 1], linewidth=2, color='dodgerblue') # changed colors and line width
    plt.plot(p_population, p_event, linewidth=2, color=line_color,
             label=(f"{model_name} Gini: {gini:.2%}")) # changed to Python 3
             
    plt.title(title, fontdict={'fontsize': 14})
    plt.xlabel(xlabel, fontdict={'fontsize': 12})
    plt.ylabel(ylabel, fontdict={'fontsize': 12})
    plt.legend(loc='lower right')

    # Save figure if requested. Pass kwargs.
    if savefig:
        plt.savefig(fname=fname, **kwargs)
        plt.close()
