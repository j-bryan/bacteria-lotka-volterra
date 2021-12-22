import numpy as np
from numpy import matmul, zeros, eye
from numpy.linalg import inv
from scipy.integrate import solve_ivp
from copy import deepcopy
from __helpers__ import get_x


class ExtendedKalmanFilter:

    """ Implements a continuous-discrete Extended Kalman filter. This is meant to serve only as a base class for a more
    specific model. As such, model-specific functions (f, FJacobian, h, HJacobian) must be implemented in that derived
    class. Users are also encouraged to overwrite the propagation methods (propagate_x and propagate_P) if the default
    implementation is not providing sufficiently accurate predictions.

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

    x : numpy.array(dim_x, 1)
        state vector

    P : numpy.array(dim_x, dim_x)
        covariance matrix

    Q : numpy.array(dim_x, dim_x)
        process noise covariance matrix

    R : numpy.array(dim_y, dim_y)
        measurement noise covariance matrix

    y : numpy.array(dim_y, 1)
        observation vector; last measurement used in update

    K : numpy.array(dim_x, dim_y)
        Kalman gain matrix - READ ONLY
    """

    def __init__(self, dim_x, dim_y):
        # state vector dimensions
        self.dim_x = dim_x
        self.dim_y = dim_y

        # The following class variables are initialized with appropriate shapes based on dim_x and dim_y; however, these
        # variables must be manually changed by the user for proper functionality

        # state vector and filter covariance
        self.x = zeros((dim_x, 1))
        self.P = eye(dim_x)

        # noise term covariances
        self.Q = eye(dim_x)
        self.R = eye(dim_y)

        # observation vector
        self.y = zeros((dim_y, 1))

        # Kalman gain
        self.K = zeros(self.x.shape)

        # copies of propagation step scipy.integrate.solve_ivp objects
        # x_prop contains solution's time array and corresponding arrays for each state vector variable
        self.x_prop_obj = None
        # P_prop contains the solution's time array and P matrices
        self.P_prop_obj = None

    def f(self, t, x):

        """ MUST BE IMPLEMENTED BY USER

        ODE/ODE system used to propagate the estimate of the mean. This should be the same function f(x(t), t)
        as used in the system model """

        raise NotImplementedError

    def FJacobian(self, x, *args):

        """ MUST BE IMPLEMENTED BY USER

        Function which returns the value of the Jacobian of f(x(t), t) evaluated at (x_k, t_k) to approximate the value
        of the state transition matrix. Used in propagating covariance matrix P

        :param x: (numpy.array(dim_x, 1)) current state vector
        """

        raise NotImplementedError

    def h(self, x, *args):

        """ MUST BE IMPLEMENTED BY USER

        Function that returns the measurement corresponding to a given system state x """

        raise NotImplementedError

    def HJacobian(self, x, *args):

        """ MUST BE IMPLEMENTED BY USER

        Function which returns the value of the Jacobian of h(x(t)) evaluated at (x_k, t_k) to approximate the value of
        measurement function matrix H """

        raise NotImplementedError

    def dP_dt(self, t, P):

        """ Continuous-time filter error propagation ODE

        :param t: (numpy.array) time array
        :param P: (numpy.array(dim_x, dim_x)) filter error covariance
        :return: numpy.array(dim_x, dim_x) dP/dt
        """

        P = np.reshape(P, (self.dim_x, self.dim_x))
        x = get_x(self.x_prop_obj, t)
        F = self.FJacobian(x)

        return matmul(F, P) + matmul(P, F.T) + self.Q

    def propagate_x(self, x, t_eval):

        """ Propagates the state vector from time t0 to time tf (given as the first and last values in array t_eval) by
        solving the ODE system dx/dt = f(x(t), t) using scipy.integrate.solve_ivp using and RK45 method. If the
        propagation of x is not sufficiently accurate for the model being used, this function should be overwritten.

        ----------
        Parameters
        ----------
        :param x : state vector before propagation
        :param t_eval : array of times [t0, ... , tf] that dictate the time steps for solve_ivp """

        def func(t, x):
            return np.ravel(self.f(t, x))

        # generate solution across interval using solve_ivp
        sol = solve_ivp(fun=func,
                        t_span=(t_eval[0], t_eval[-1]),
                        y0=np.ravel(self.x),
                        t_eval=t_eval)

        # save solution object
        self.x_prop_obj = deepcopy(sol)

        # assemble a new x vector from the last values of each variable from sol.y
        new_x = []
        for yi in sol.y:
            new_x.append([yi[-1]])

        return np.array(new_x)

    def propagate_P(self, t_eval):

        """ Propagates the filter error covariance matrix P from t0 to tf by solving the ODE system
                    dP/dt = FP + PF^T + Q
         where F is the Jacobian of f(x(t), t) evaluated at the current x. Uses _________ to solve the IVP.

        ----------
        Parameters
        ----------
        :param t_eval : array of times [t0, ... , tf] that dictate the time steps for solve_ivp
        """

        def func(t, x):
            return np.ravel(self.dP_dt(t, x))

        # solve dP/dt IVP
        sol = solve_ivp(fun=func,
                        t_span=(t_eval[0], t_eval[-1]),
                        y0=np.ravel(self.P),
                        t_eval=t_eval)

        P_arr = []

        for i in range(len(sol.t)):
            P = []
            for j in range(len(sol.y)):
                P.append(sol.y[j][i])
            P = np.array(P)
            P_arr.append(np.reshape(P, self.P.shape))

        sol.y = P_arr

        self.P_prop_obj = deepcopy(sol)

        return self.P_prop_obj.y[-1]

    def propagate(self, t0, tf, steps):

        """ Propagates the filter mean estimation and error covariance forward from t0 to tf using the propagate_x
        and propagate_P methods, respectively """

        # create time interval array from t0, tf, and number of elements in the array
        t_eval = np.linspace(t0, tf, steps)

        # save a copy of x to pass into propagate_P after x is updated
        x0 = self.x

        # propagate x and P
        self.x = self.propagate_x(self.x, t_eval)
        self.P = self.propagate_P(t_eval)

    def update(self, y, H_args=()):

        """ Updates the filter with observation y

        :param y: observation vector; should be of shape (dim_y, 1)
        :param H_args: (optional) additional arguments to pass to HJacobian method
        """

        # verify H_args is a correctly-formatted tuple
        if not isinstance(H_args, tuple):
            H_args = (H_args, )

        # save y vector
        self.y = deepcopy(y)

        # approximate a linearization of h(x(t), t) using HJacobian
        H = self.HJacobian(self.x, *H_args)

        # calculate Kalman gain
        PHT = matmul(self.P, H.T)  # used twice when calculating K;
        self.K = matmul(PHT, inv(matmul(H, PHT) + self.R))

        # update x and P
        self.x = self.x + matmul(self.K, y - self.h(self.x))
        self.P = matmul(eye(self.dim_x) - matmul(self.K, H), self.P)

    def print_state(self):
        print("Extended Kalman Filter Report")
        print("x:", self.x)
        print("P:", self.P)
        print("Q:", self.Q)
        print("R:", self.R)

    @property
    def check_x(self):
        if self.x is None:
            self.print_state()
        return self.x
