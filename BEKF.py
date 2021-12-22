from filters import ExtendedKalmanFilter
import numpy as np


class BacteriaEKF(ExtendedKalmanFilter):

    """ An implementation of a two-dimensional Lotka-Volterra competition model that uses an Extended Kalman Filter to
    estimate the true population density of each species being tracked when accounting for the ideal model prediction
    and observations of the population.

    ----------------
    -- Parameters --
    ----------------

    dim_x : int
        dimension of state vector x

    dim_y : int
        dimension of observation vector y

    ----------------
    -- Attributes --
    ----------------

    g : numpy.array(dim_x)
        intrinsic growth rate of each species

    c : numpy.array(dim_x, dim_x)
        interspecific competition factors

    """

    def __init__(self, dim_x, dim_y):
        # initialize ExtendedKalmanFilter class
        super().__init__(dim_x, dim_y)

        # initialize g and c
        self.g = np.ones(dim_x)
        self.c = np.ones((dim_x, dim_x))

    def f(self, t, x):

        """ ODE/ODE system used to propagate the estimate of the mean. This should be the same function f(x(t), t) as
        used in the system model """

        x0, x1 = np.ravel(x)
        g = self.g
        c = self.c

        return np.array([[x0 * (g[0] + c[0][0]*x0 + c[0][1]*x1)],
                         [x1 * (g[1] + c[1][0]*x0 + c[1][1]*x1)]])

    def FJacobian(self, x, *args):

        """ Function which returns the value of the Jacobian of f(x(t), t) evaluated at (x_k, t_k) to approximate the
        value of the state transition matrix. Used in propagating covariance matrix P """

        # if x is None:
        #     print(x)
        #     print(self.x)
        #     print()

        # x0, x1 = np.ravel(x)
        x0, x1 = np.ravel(self.x)
        g = self.g
        c = self.c

        return np.array([[g[0] + 2*c[0][0]*x0 + c[0][1]*x1, c[0][1]*x0],
                         [c[1][0]*x1,                       g[1] + c[1][0]*x0 + 2*c[1][1]*x1]])

    def h(self, x, *args):

        """ Function that returns the measurement corresponding to a given system state x """

        return x

    def HJacobian(self, x, *args):

        """ Function which returns the value of the Jacobian of h(x(t)) evaluated at (x_k, t_k) to approximate the value
        of measurement function matrix H """

        return np.eye(self.dim_x)
