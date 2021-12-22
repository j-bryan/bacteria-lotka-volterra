import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)


def centered_finite_diff(x, y):
    ret_x = np.zeros(len(x) - 1)
    ret_y = np.zeros(len(y) - 1)
    for i in range(len(x) - 1):
        ret_x[i] = (x[i + 1] + x[i]) / 2
        ret_y[i] = (y[i + 1] - y[i]) / (x[i + 1] - x[i])
    return ret_x, ret_y


def trapezoidal(x, y):
    ret_x = np.zeros(len(x) - 1)
    ret_y = np.zeros(len(y) - 1)
    for i in range(len(x) - 1):
        ret_x[i] = (x[i + 1] + x[i]) / 2
        ret_y[i] = (x[i + 1] - x[i]) * (y[i] + y[i + 1]) / 2
    return ret_x, ret_y


def tvr_diff(x, f, alpha, eps=0.1, maxiter=1000, emax=1e-6):
    """
    Uses a total variation regularized derivative to numerically differentiate noisy data. This data is given as an
    array of n-tuples or subarrays. This method accommodates discontinuous derivatives.

    :param x: (numpy.ndarray) x points
    :param f: (numpy.ndarray) y points -- these are noisy
    :param alpha: regularization parameter; should be chosen such that the mean-squared difference between Au and f
                    should equal the variance of the noise of f
    :param maxiter: maximum number of iterations to perform in denoising process
    :param emax: maximum allowable difference between iterations
    """
    reshaped_f = False
    if len(f.shape) == 1:
        f = f.reshape((-1, 1))
        reshaped_f = True
    elif len(f.shape) != 2:
        raise ValueError('Your f array is a funny shape. Try again, dude.')

    n = int(len(f) - 1)
    dx = (x[-1] - x[0]) / n  # assumes constant spacing
    yhat = f - np.full((n + 1, 1), f[0])

    # generate a random vector for a first guess of un
    un = np.zeros((n + 1, 1))

    # generate the differentiation matrix D
    D = np.zeros((n, n + 1))
    one_over_dx = 1 / dx
    for i in range(n):
        D[i][i] = -one_over_dx
        D[i][i + 1] = one_over_dx

    # generate the antidifferentiation matrix K
    K = np.zeros((n + 1, n + 1))
    for i in range(1, n + 1):
        K[i] = K[i - 1]
        K[i][i - 1] += 1
        K[i][i] += 1
    K *= (x[-1] - x[0]) / (2 * n)

    it = 0
    while it < maxiter:
        # generate E matrix
        En = np.zeros((n, n))
        for i in range(n):
            En[i][i] = ((un[i + 1] - un[i]) ** 2 + eps) ** -0.5

        # calculate L matrix
        Ln = dx * np.dot(D.T, En).dot(D)

        # calculate H matrix
        Hn = np.dot(K.T, K) + alpha * Ln

        # calculate g
        gn = np.dot(K.T, np.dot(K, un) - yhat) + alpha * np.dot(Ln, un)

        # calculate s
        sn = -1 * np.dot(np.linalg.inv(Hn), gn)

        # solve for u_(n+1)
        un += sn

        it += 1
        # check if sn values are all small --> u reached stationarity
        if np.all(abs(sn) < emax):
            break

    if reshaped_f:
        un = np.ravel(un)
    return un


if __name__ == '__main__':
    n = 100
    x = np.linspace(0, 1, n)
    y = np.abs(x - 0.5)
    f = y + np.random.normal(0, 0.05, n)

    u = tvr_diff(x, y, alpha=0.001, eps=1e-6)

    K = np.zeros((n, n))
    for i in range(1, n):
        K[i] = K[i - 1]
        K[i][i - 1] += 1
        K[i][i] += 1
    K *= (x[-1] - x[0]) / (2 * n)

    plt.figure(1)
    plt.plot(x, y, color='k', label='true function g')
    plt.plot(x, K.dot(u) + y[0], color='g', linestyle='--', label='computed function f')
    plt.scatter(x, f, marker='o', edgecolors='red', facecolors='none', label='data points')
    plt.legend()

    plt.figure(2)
    plt.plot(x, [-1 if xi < 0.5 else 1 for xi in x], color='k', label='true derivative g\'')
    plt.plot(x, u, color='g', linestyle='--', label='calculated u')
    plt.legend()

    plt.show()
