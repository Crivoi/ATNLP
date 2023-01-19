import math
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def show_plot(points):
    plt.figure()
    fig, ax = plt.subplots()
    ax.plot(points)
    fig.savefig(f'./plots/plot_loss_{len(points)}_iters.png')
    plt.show()


def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))


def bar_plot(splits, mean_results, std_results, x_label='Ground-truth action seq length',
             y_label='Accuracy on new commands (%)', title='Sequence Length'):
    x_pos = np.arange(len(splits))
    fig, ax = plt.subplots()
    ax.bar(x_pos, list(mean_results.values()), align='center',
           yerr=list(std_results.values()), ecolor='black', alpha=.5)
    ax.set_xlabel(xlabel=x_label)
    ax.set_ylabel(ylabel=y_label)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(splits)
    plt.ylim((0., 1.))
    plt.savefig(f'./plots/{title}_plot.png')
    plt.show()
