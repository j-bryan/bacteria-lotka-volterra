import numpy as np
from utils import jacobian, rk4


class AugStateEKF:

    """ Augmented state extended Kalman filter with additional system parameters appended to the state vector"""

    def __init__(self, dim_x, dim_theta, dim_y):
        assert isinstance(dim_x, int) and dim_x > 0
        assert isinstance(dim_theta, int) and dim_theta > 0
        assert isinstance(dim_y, int) and dim_y > 0

        self.dim_x = dim_x
        self.dim_theta = dim_theta
        self.dim_y = dim_y
        self.dim_z = dim_x + dim_theta

        self.z = np.zeros((self.dim_z, 1))
        self.P = np.eye(self.dim_z)
        self.y = np.zeros((dim_y, 1))
        self.K = np.zeros((dim_y, self.dim_z))

        self.t_prop = None
        self.z_prop = None
        self.P_prop = None

    def propagate(self, t0, tf, steps=101):
        self.t_prop, self.z_prop, self.P_prop = rk4(self.f, self.dPdt, self.z, self.P, t0, tf, steps)
        self.z = self.z_prop[-1]
        self.P = self.P_prop[-1]
        return self.t_prop, self.z_prop, self.P_prop

    def update(self, t, y, R):
        self.y = y.reshape(self.y.shape)

        H = jacobian(self.h, t, self.z)
        PHT = np.dot(self.P, H.T)
        self.K = np.dot(PHT, np.linalg.inv(np.dot(H, PHT) + R))

        self.z = self.z + np.dot(self.K, self.y - self.h(t, self.z))
        self.P = np.dot(np.eye(self.dim_z) - np.dot(self.K, H), self.P)

        return t, self.z, self.P

    def f(self, t, z):
        raise NotImplementedError

    def h(self, t, z):
        raise NotImplementedError

    def dPdt(self, t, P, z):
        F = jacobian(self.f, t, z)
        return np.dot(F, P) + np.dot(P, F.T)

    @property
    def x(self):
        return self.z[:self.dim_x]

    @x.setter
    def x(self, val):
        assert val.shape == (self.dim_x, 1)
        self.z[:self.dim_x] = val

    @property
    def theta(self):
        return self.z[self.dim_x:]

    @theta.setter
    def theta(self, val):
        assert val.shape == (self.dim_theta, 1)
        self.z[self.dim_x:] = val
