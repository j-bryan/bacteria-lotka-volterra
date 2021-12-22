import numpy as np


def rk4(f, x0, t_range, steps):
    assert len(t_range) == 2

    h = (t_range[1] - t_range[0]) / (steps - 1)
    ts = np.linspace(t_range[0], t_range[1], steps)
    xs = np.zeros((len(ts), len(x0)))
    xs[0] = x0

    for i in range(steps - 1):
        k1 = h * f(xs[i], ts[i])
        k2 = h * f(xs[i] + k1 / 2, ts[i] + h / 2)
        k3 = h * f(xs[i] + k2 / 2, ts[i] + h / 2)
        k4 = h * f(xs[i] + k3, ts[i] + h)
        xs[i + 1] = xs[i] + (k1 + 2 * (k2 + k3) + k4) / 6

    return ts, xs
