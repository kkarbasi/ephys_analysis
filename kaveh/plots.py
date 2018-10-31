import pylab as pl
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import collections
from matplotlib import transforms

def axvlines(ax, x, **kw):
    """
    Adds vertical lines to ax at horizontal positions in x
    """
    from matplotlib import collections
    from matplotlib import transforms

    x = np.asanyarray(x)
    y0 = np.zeros_like(x)
    y1 = np.ones_like(x)
    data = np.c_[x, y0, x, y1].reshape(-1, 2, 2)
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    lines = collections.LineCollection(data, transform=trans, **kw)
    ax.add_collection(lines)

def plot_raster_overlaid_signal(signal_vals, t, spike_times, **kw):
    """
    Plots signal_vals vs t, overlaid by a raster plot of spike_times
    """
    plt.figure()
    plt.plot(t, signal_vals, zorder=1)
    ax = plt.gca()
    axvlines(ax, spike_times, **kw)
    plt.show()
    # plt.title('{}'.format(asig.annotations['channel_names'][0]))
    # plt.ylabel('{}'.format(str(asig.units)))
    # plt.xlabel('t (s)')
    return ax

def plot_shaded_err(xf, mean_signal, err_signal, **kw):
    """
    Plots the average spike wavelets of the current dataset
    """
    l = plt.plot(x, mean_signal, **kw)
    plt.fill_between(x, mean_signal - err_signal, mean_signal + err_signal, color=l[0].get_color(), alpha=0.25) 
    