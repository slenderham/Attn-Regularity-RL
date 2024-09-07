import jax.numpy as jnp
import numpy as np
from numpyro.diagnostics import hpdi


def plot_mean_hpdi(ax, x, color, label, axis=0, prob=0.95, alpha=0.2):
    mx = jnp.nanmean(x, axis)
    ax.plot(mx, c=color, label=label, lw=3)
    x_hpdi = hpdi(x, prob, axis)
    ax.fill_between(jnp.arange(mx.shape[-1]), x_hpdi[0], x_hpdi[1], alpha=alpha, color=color)

def nanmovmean(x, window_size, axis=0):
    x_len = x.shape[axis]
    smth_x = []
    for t in range(x_len-window_size):
        smth_x.append(np.nanmean(x.take(jnp.arange(t,t+window_size), axis=axis), axis=axis))
    return np.stack(smth_x, axis=axis)