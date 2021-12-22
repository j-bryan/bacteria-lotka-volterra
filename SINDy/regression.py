import numpy as np
from SINDy.library import gen_lib, gen_callable_model
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


def regression(X, dXdt, thresh, poly_deg=0, trig_deg=0, exp_deg=0, it=10):
    Theta = gen_lib(X, poly_deg=poly_deg, trig_deg=trig_deg, exp_deg=exp_deg)
    Xi = np.linalg.lstsq(Theta, dXdt, rcond=None)[0]
    threshold(Xi, Theta, dXdt, thresh, it)
    return Xi, Theta


def discrete_regression(X, thresh, poly_deg=0, trig_deg=0, exp_deg=0, it=10):
    X1 = X[:-1, ]
    X2 = X[1:, ]
    Theta = gen_lib(X1, poly_deg=poly_deg, trig_deg=trig_deg, exp_deg=exp_deg)
    Xi = np.linalg.lstsq(Theta, X2, rcond=None)[0]
    threshold(Xi, Theta, X2, thresh, it)
    return Xi, Theta


def lstsq_regression(X, dXdt):
    C = X[:, 0]
    D = X[:, 1]
    Theta = np.array([np.ones(len(C)), C, D, C ** 2, C * D, D ** 2]).T
    Xi = np.zeros((6, 2))
    Xi[[1, 3, 4], 0] = np.linalg.lstsq(Theta[:, [1, 3, 4]], dXdt[:, 0], rcond=None)[0]
    Xi[[2, 4, 5], 1] = np.linalg.lstsq(Theta[:, [2, 4, 5]], dXdt[:, 1], rcond=None)[0]
    return Xi, Theta


def linearCD(X, dXdt):
    C = X[:, 0]
    D = X[:, 1]
    Theta = np.array([C,
                      D,
                      C ** 2,
                      C * D,
                      D ** 2,
                      C ** 2 / (C + D),
                      C * D / (C + D),
                      C ** 3 / (C + D),
                      C ** 2 * D / (C + D),
                      C * D ** 2 / (C + D)]).T
    Xi = np.zeros((10, 2))
    xi1_nzi = [0, 2, 3, 5, 7, 8]
    xi2_nzi = [1, 3, 4, 6, 8, 9]
    Xi[xi1_nzi, 0] = np.linalg.lstsq(Theta[:, xi1_nzi], dXdt[:, 0], rcond=None)[0]
    Xi[xi2_nzi, 1] = np.linalg.lstsq(Theta[:, xi2_nzi], dXdt[:, 1], rcond=None)[0]
    return Xi, Theta


def linC_quadD(X, dXdt):
    C = X[:, 0]
    D = X[:, 1]
    Theta = np.array([C,
                      D,
                      C ** 2,
                      C * D,
                      D ** 2,
                      C ** 2 / (C + D),
                      C * D / (C + D),
                      C ** 3 / (C + D),
                      C ** 2 * D / (C + D),
                      C * D ** 2 / (C + D),
                      C ** 2 * D / (C + D) ** 2,
                      C ** 3 * D / (C + D) ** 2,
                      C ** 2 * D ** 2 / (C + D) ** 2]).T
    Xi = np.zeros((13, 2))
    xi1_nzi = [0, 2, 3, 5, 7, 8, 11]
    xi2_nzi = [1, 3, 4, 6, 8, 9, 10, 12]
    Xi[xi1_nzi, 0] = np.linalg.lstsq(Theta[:, xi1_nzi], dXdt[:, 0], rcond=None)[0]
    Xi[xi2_nzi, 1] = np.linalg.lstsq(Theta[:, xi2_nzi], dXdt[:, 1], rcond=None)[0]
    return Xi, Theta


def quadC_linD(X, dXdt):
    C = X[:, 0]
    D = X[:, 1]
    Theta = np.array([C,
                      D,
                      C ** 2,
                      C * D,
                      D ** 2,
                      C ** 2 / (C + D),
                      C * D / (C + D),
                      C ** 3 / (C + D),
                      C ** 2 * D / (C + D),
                      C * D ** 2 / (C + D),
                      C ** 3 / (C + D) ** 2,
                      C ** 4 / (C + D) ** 2,
                      C ** 3 * D / (C + D) ** 2]).T
    Xi = np.zeros((13, 2))
    xi1_nzi = [0, 2, 3, 5, 7, 8, 10, 11]
    xi2_nzi = [1, 3, 4, 6, 8, 9, 12]
    Xi[xi1_nzi, 0] = np.linalg.lstsq(Theta[:, xi1_nzi], dXdt[:, 0], rcond=None)[0]
    Xi[xi2_nzi, 1] = np.linalg.lstsq(Theta[:, xi2_nzi], dXdt[:, 1], rcond=None)[0]
    return Xi, Theta


def quadCD(X, dXdt):
    C = X[:, 0]
    D = X[:, 1]
    Theta = np.array([C,
                      D,
                      C ** 2,
                      C * D,
                      D ** 2,
                      C ** 2 / (C + C),
                      C * D / (C + D),
                      C ** 3 / (C + D),
                      C ** 2 * D / (C + D),
                      C * D ** 2 / (C + D),
                      C ** 3 / (C + D) ** 2,
                      C ** 2 * D / (C + D) ** 2,
                      C ** 4 / (C + D) ** 2,
                      C ** 3 * D / (C + D) ** 2,
                      C ** 2 * D ** 2 / (C + D) ** 2]).T
    Xi = np.zeros((15, 2))
    xi1_nzi = [0, 2, 3, 5, 7, 8, 10, 12, 13]
    xi2_nzi = [1, 3, 4, 6, 8, 9, 11, 13, 14]
    Xi[xi1_nzi, 0] = np.linalg.lstsq(Theta[:, xi1_nzi], dXdt[:, 0], rcond=None)[0]
    Xi[xi2_nzi, 1] = np.linalg.lstsq(Theta[:, xi2_nzi], dXdt[:, 1], rcond=None)[0]
    return Xi, Theta


def threshold(Xi, Theta, dXdt, thresh, it):
    for _ in range(it):
        Xi[np.abs(Xi) < thresh] = 0
        for i in range(Xi.shape[1]):
            nonzero_inds = np.argwhere(np.abs(Xi[:, i]) > thresh)
            new_Xi_entries = np.linalg.lstsq(Theta[:, np.ravel(nonzero_inds)], dXdt[:, i], rcond=None)[0]
            Xi[nonzero_inds, i] = new_Xi_entries.reshape((-1, 1))


def regression_test():
    """ Damped oscillator with linear terms. Duplication of Example 1a from "Discovering governing equations from data"
    by Brunton, et al"""

    # function version for solve_ivp
    def dXdt_func(t, x):
        X = np.array([x]).T
        mat = np.array([[-0.1, 2],
                        [-2, -0.1]])
        return np.dot(mat, X).T[0]

    # function version used for calculating dXdt for regression function
    def dXdt_for_realsies(X):
        mat = np.array([[-0.1, 2],
                        [-2, -0.1]])
        return np.dot(mat, X.T).T

    sol = solve_ivp(dXdt_func, y0=[2, 0], t_span=[0, 25], t_eval=np.linspace(0, 25, 1000))  # get X measurements by using a RKF method
    t = sol.t
    X = np.array(sol.y).T
    X += 0.1 * np.random.standard_normal(size=X.shape)  # add noise to X data
    plt.plot(t, sol.y[0], color='red')
    plt.plot(t, sol.y[1], color='blue')
    dXdt = dXdt_for_realsies(X)
    dXdt += 0.1 * np.random.standard_normal(size=dXdt.shape)  # add noise to dXdt data
    Xi, Theta = regression(X, dXdt, thresh=4e-2, poly_deg=5)  # run regression; NOTE: very sensitive to threshold value!
    print(Xi)
    plt.show()


def discrete_regression_test():
    """ Damped oscillator with linear terms. Similar to Example 1a from "Discovering governing equations from data"
        by Brunton, et al, but using their discrete-time sparse regression implementation (Section 3.3.1) """

    def dXdt_func(t, x):
        X = np.array([x]).T
        mat = np.array([[-0.1, 2],
                        [-2, -0.1]])
        return np.dot(mat, X).T[0]

    def dXdt_true(X):
        mat = np.array([[-0.1, 2],
                        [-2, -0.1]])
        return np.dot(mat, X.T).T

    sol = solve_ivp(dXdt_func, y0=[2, 0], t_span=[0, 25], t_eval=np.linspace(0, 25, 1000))
    X = np.array(sol.y).T
    X += 0.1 * np.random.standard_normal(size=X.shape)
    dXdt = dXdt_true(X)
    dXdt += 0.1 * np.random.standard_normal(size=dXdt.shape)
    Xi, Theta = regression(X, dXdt, thresh=2e-2, poly_deg=5)
    print(Xi)
