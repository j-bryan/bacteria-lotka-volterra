import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from utils.diff import centered_finite_diff


def least_squares(x, y, deg, ret_func=False):
    """ Finds a least squares fit to the data, then takes the derivative of that polynomial fit. """
    coeffs = np.polyfit(x, y, deg)
    diff_func = np.poly1d([(deg - i) * coeffs[i] for i in range(deg)])
    if ret_func:
        func = np.poly1d(coeffs)
        return func(x), diff_func(x)
    else:
        return diff_func(x)


def loess():
    pass


def savitsky_golay(y, window, order):
    """ Savitsky-Golay smoothing filter

    Not good with high levels of noise or very dense data! Better with sparse data w/ small noise.
    """
    return savgol_filter(y, window, order)


def tikhonov(x, y, alpha, k):
    """ Tikhonov regularization """
    n = len(x) - 1
    yhat = y - y[0]

    # switching to t uniform grid from x makes it so that x doesn't have to be uniform
    dt = (x[-1] - x[0]) / n
    t = np.array([x[0] + j * dt for j in range(n + 1)])  # (n + 1)-dimensional (same as x and y)

    # u is defined on the midpoints on the uniform t grid (n-dimensional)
    u = np.zeros(n)

    # define A matrix in Au = y - y[0]
    # A = np.zeros((n + 1, n))
    # for i in range(len(x)):
    #     for j in range(len(t) - 1):
    #         if t[j] < x[i] < t[j + 1]:
    #             A[i][j] = x[i] - t[j]
    #         elif t[j + 1] <= x[i]:
    #             A[i][j] = dt
    A = np.zeros((n, n + 1))
    A[0][0] = dt
    A[0][1] = dt
    for i in range(1, n):
        A[i] = A[i - 1]
        A[i][i] += dt
        A[i][i + 1] += dt
    A = A.T

    # # define differential operator D. Larger values of k enforce more smoothness
    # def Dk(m):
    #     D_mat = np.zeros((m, m - 1))
    #     for i in range(m):
    #         D_mat[i][i] = -1
    #         D_mat[i][i + 1] = 1
    #     return 1 / dt * D_mat
    #
    # D1 = Dk(n)  # maps midpoint values to interior grid points
    # D2 = Dk(n - 1)  # maps midpoint values to interior midpoint values
    #
    # D = np.eye(n)  # k == 0
    # if k == 1:
    #     D = D1
    # elif k == 2:
    #     D = np.dot(D1, D2)
    # elif k != 0:
    #     raise ValueError('k must be 0, 1, or 2 for now')
    D = np.zeros((n, n + 1))
    for i in range(n):
        D[i][i] = -1
        D[i][i + 1] = 1
    D = D.T / dt

    LHS = np.dot(A.T, A) + alpha * np.dot(D.T, D)
    RHS = np.dot(A.T, yhat)
    return np.array([t[0] + (i + 0.5) * dt for i in range(n)]), np.dot(np.linalg.inv(LHS), RHS)


def cubic_spline(x, y, alpha=0.1):
    """ Cubic spline smoothing method """
    n = len(x)
    Delta = np.zeros((n - 2, n))
    W = np.zeros((n - 2, n - 2))

    for i in range(n - 2):
        Delta[i][i] = 1 / (x[i + 1] - x[i])
        Delta[i][i + 1] = -1 / (x[i + 1] - x[i]) - 1 / (x[i + 2] - x[i + 1])
        Delta[i][i + 2] = 1 / (x[i + 2] - x[i + 1])

    for i in range(n - 2):
        W[i][i] = (x[i + 2] - x[i]) / 3
        if i != 0:
            W[i - 1][i] = W[i][i - 1] = (x[i + 1] - x[i]) / 6

    A = np.dot(np.dot(Delta.T, np.linalg.inv(W)), Delta)
    mhat = np.dot(np.linalg.inv(np.eye(n) + alpha * A), y.reshape((-1, 1)))
    return np.ravel(mhat)


def cubic_spline_diff(x, y, alpha):
    # define new grid with (xi, yi) on midpoints
    # interpolate to put (x, y) on new grid
    pass


def convolution(x, y, N):
    """ Computes a smooth approximation of a function by convolution of a piecewise linear interpolant with the positive
    symmetric Friedrichs mollifier function. Abscissae must be distinct. Approximation is valid only on [a+h, b-h]. """
    box = np.ones(N) / N
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def KnW_variational():
    """ Knowles and Wallace variational method. Designed with the goal of eliminating the difficult problem of
    determining an effective regularization parameter when nothing is known about the errors in the data. """
    pass


def total_variation():
    """ Smoothing method using total variation regularization to smooth data while maintaining discontinuities. """
    pass


# TEST METHODS
def test_least_squares():
    x = np.linspace(0, 10, 100)
    y = x ** 2 + np.random.normal(loc=0, scale=5, size=len(x))
    deg = 2
    f, u = least_squares(x, y, deg=deg, ret_func=True)

    plt.subplot(121)
    plt.plot(x, x ** 2, label='True function g')
    plt.plot(x, f, label='Computed function f')
    plt.scatter(x, y, marker='o', facecolors='none', edgecolors='r', label='Data points')
    plt.legend()

    plt.subplot(122)
    plt.plot(x, 2 * x, label='True derivative g\'')
    plt.plot(x, u, label='computed solution u', linestyle='--')
    plt.legend()

    plt.show()


def test_savgol(window, order):
    x = np.linspace(-0.5, 0.5, 10)
    y = np.cos(x) + np.random.normal(loc=0, scale=0.01, size=len(x))
    f = savitsky_golay(y, window, order)

    # plt.subplot(121)
    plt.plot(x, np.cos(x), label='True function g')
    plt.plot(x, f, label='Computed function f')
    plt.scatter(x, y, marker='o', facecolors='none', edgecolors='r', label='Data points')
    plt.xlim([-0.5, 0.5])
    plt.legend()

    # plt.subplot(122)
    # plt.plot(x, -np.sin(x), label='True derivative g\'')
    # plt.plot(t, u, label='computed solution u', linestyle='--')
    # plt.xlim([-0.5, 0.5])
    # plt.legend()

    plt.show()


def test_tikhonov(alpha, k):
    x = np.linspace(-0.5, 0.5, 100)
    y = np.cos(x) + np.random.normal(loc=0, scale=0.1, size=len(x))
    t, u = tikhonov(x, y, alpha, k)

    plt.subplot(121)
    plt.plot(x, np.cos(x), label='True function g')
    # plt.plot(x, f, label='Computed function f')
    plt.scatter(x, y, marker='o', facecolors='none', edgecolors='r', label='Data points')
    plt.xlim([-0.5, 0.5])
    plt.legend()

    plt.subplot(122)
    plt.plot(x, -np.sin(x), label='True derivative g\'')
    plt.plot(t, u, label='computed solution u', linestyle='--')
    plt.xlim([-0.5, 0.5])
    plt.legend()

    plt.show()


def test_cubic_spline(N, sigma, alpha):
    x = np.linspace(-0.5, 0.5, N)
    y = np.cos(x) + np.random.normal(loc=0, scale=sigma, size=len(x))
    # f, xdiff, u = cubic_spline(x, y, alpha=0.01, ret_func=True)
    f = cubic_spline(x, y, alpha=alpha)

    plt.subplot(121)
    plt.plot(x, np.cos(x), label='True function g')
    plt.plot(x, f, label='Computed function f')
    plt.scatter(x, y, marker='o', facecolors='none', edgecolors='r', label='Data points')
    plt.xlim([-0.5, 0.5])
    plt.legend()

    plt.subplot(122)
    plt.plot(x, -np.sin(x), label='True derivative g\'')
    plt.plot(*centered_finite_diff(x, f), label='computed solution u', linestyle='--')
    plt.xlim([-0.5, 0.5])
    plt.legend()

    plt.show()


def test_convolution():
    x = np.linspace(-0.5, 0.5, 100)
    y = np.cos(x) + np.random.normal(loc=0, scale=0.01, size=len(x))
    f = convolution(x, y, N=19)

    # plt.subplot(121)
    plt.plot(x, np.cos(x), label='True function g')
    plt.plot(x, f, label='Computed function f')
    plt.scatter(x, y, marker='o', facecolors='none', edgecolors='r', label='Data points')
    plt.xlim([-0.5, 0.5])
    plt.legend()

    # plt.subplot(122)
    # plt.plot(x, -np.sin(x), label='True derivative g\'')
    # plt.plot(xdiff, u, label='computed solution u', linestyle='--')
    # plt.xlim([-0.5, 0.5])
    # plt.legend()

    plt.show()


if __name__ == '__main__':
    # test_least_squares()  # looks good
    # test_savgol(5, 3)
    # test_tikhonov(x, y, alpha=0.00078507, k=2)  # TODO: tikhonov isn't working...
    test_cubic_spline(N=10, sigma=0.01, alpha=0.005)
    # test_convolution()
