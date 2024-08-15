import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import freqz


def plot_filters(filters, titles, colors, window_size):
    #print('Titles (shape):', titles.shape)
    #print('Titles:', titles)
    #print('Filters:', filters.shape)

    # Create a figure
    fig = plt.figure(figsize=(13, 6), dpi=100, num="Bandpass Filter Frequency Response")
    fig.subplots_adjust(hspace=0.5)

    gs = gridspec.GridSpec(3, 5)  # 3 rows, 5 columns

    # Create subplots in specific grid locations
    subplot = [None] * len(filters)
    filter_sum = np.zeros(len(filters[0]))
    for i in range(len(filters)):
        if i <= 4:
            subplot[i] = fig.add_subplot(gs[0, i])
        else:
            subplot[i] = fig.add_subplot(gs[1, i - 5])
        filter_sum += filters[i]

    all_filters = fig.add_subplot(gs[2, :4])  # subplot spans all columns of the third row
    subplot_filter_sum = fig.add_subplot(gs[2, 4:])  # subplot spans all columns of the third row

    # Example data for plotting
    x = np.arange(window_size)
    for i in range(len(filters)):
        w, h = freqz(filters[i], 1.0, worN=window_size)
        x = w / np.pi
        y = 20 * np.log10(np.abs(h))
        subplot[i].set_title(titles[i])
        subplot[i].plot(x, y, colors[i])
        all_filters.plot(x, y, colors[i])
    
    subplot_filter_sum.set_title("Filters sum")
    subplot_filter_sum.plot(x, filter_sum)

    plt.show()


# Code for test
if __name__ == "__main__":
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    frequences = ['32Hz', '64Hz', '128Hz', '256Hz', '512Hz', '1KHz', '2KHz', '4KHz', '8KHz', '16KHz']
    x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    ax = [None] * 10
    ax[0] = [i for i in x]
    ax[1] = [i for i in x]
    ax[2] = [i for i in x]
    ax[3] = [i**2 for i in x]
    ax[4] = [i**0.5 for i in x]
    ax[5] = [i for i in x]
    ax[6] = [i for i in x]
    ax[7] = [i for i in x]
    ax[8] = [i**2 for i in x]
    ax[9] = [i**0.5 for i in x]
    plot_filters(ax, frequences, colors)
