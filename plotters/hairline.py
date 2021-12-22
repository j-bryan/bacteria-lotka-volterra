import numpy as np
from numpy.random import multivariate_normal as mv_norm
import matplotlib.pyplot as plt
from __helpers__ import find
from copy import deepcopy
from scipy.integrate import solve_ivp


class Hairline:
    """ Creates a hairline plot in order to show the effectiveness of an Extended Kalman Filter

    ----------------
    -- Parameters --
    ----------------

    ekf_obj : filters.EKF object
        contains an initialized extended Kalman filter

    num_samples : int
        number of hairline samples to create and plot

    times : numpy.array
        numpy array of times over which the filter is run and observations are generated

    prop_steps : int


    update_freq : int
        number of propagation steps to perform between updates

    """

    def __init__(self, ekf_obj, num_samples, times, prop_steps, update_freq):
        self.ekf_init = deepcopy(ekf_obj)
        self.ekf = deepcopy(ekf_obj)
        self.num_samples = num_samples
        self.t = times
        self.update_freq = update_freq
        self.prop_steps = prop_steps
        self.meas = None

        self.x0_mean = None
        self.x1_mean = None

    def __gen_measurements(self, t_eval, x0):
        """ Generates synthetic observations from the system model plus a noise term

        :param t_eval: (numpy.array) array of timesteps
        :param x0: (numpy.array(dim_x, 1))
        :return: solution times (numpy.array), x0 solution (numpy.array), x1 solution (numpy.array)
        """
        print("Generating observations")

        # wrap self.ekf.f to work with solve_ivp
        def func(t, x):
            # noise = mv_norm(np.zeros(self.ekf.dim_x), self.ekf.R)
            # return np.ravel(self.ekf.f(t, x)) + 5 * noise
            return np.ravel(self.ekf.f(t, x))

        # use solve_ivp to produce measurements at all times in t_eval
        sol = solve_ivp(fun=func,
                        t_span=[t_eval[0], t_eval[-1]],
                        y0=x0,
                        t_eval=t_eval)

        return [sol.t, sol.y[0], sol.y[1]]

    def __run_filter(self, filter_obj, times, update_freq, prop_steps, measurements):
        """ Run filter instance across times array """

        # initialize some useful variables/storage arrays
        bekf = filter_obj
        t_meas, x0_meas, x1_meas = measurements

        t_filter = []
        x0_filter = []
        x1_filter = []
        P_filter = []

        # each loop covers the span between two adjacent update times
        for i in range(len(times) // update_freq + 1):
            # calculate time interval start (t0) and stop (tf)
            t0 = times[update_freq * i]

            # this makes sure the end of the times array is used even if the array would be unevenly subdivided by the
            # provided update frequency
            if update_freq * (i + 1) > len(times) - 1:
                tf = times[-1]
            else:
                tf = times[update_freq * (i + 1)]

            if t0 == tf:
                break

            # propagate the filter and get the t, x0, x1, and P arrays from that propagation interval
            bekf.propagate(t0, tf, prop_steps)

            t_filter.extend(bekf.x_prop_obj.t)
            x0_filter.extend(bekf.x_prop_obj.y[0])
            x1_filter.extend(bekf.x_prop_obj.y[1])
            P_filter.extend(bekf.P_prop_obj.y)

            # find the appropriate measurement in the synthetic observation arrays and update the filter with those
            # observations
            meas_index = find(t_meas, tf)
            y = np.array([[x0_meas[meas_index]],
                          [x1_meas[meas_index]]])
            bekf.update(y)

        return t_filter, x0_filter, x1_filter, P_filter

    def __sample(self, mean, cov):
        """ generates self.num_samples Monte Carlo samples of the initial state distribution """
        mean = np.ravel(mean)
        samples = []
        for i in range(self.num_samples):
            samples.append(mv_norm(mean, cov))
        return samples

    def __get_filter_cov(self, Ps):
        """ calculates the mean covariance of the filter from the covariance matrices (P) of each MC sample """
        P_mean = np.zeros((len(Ps[0]), 2, 2))  # initialize mean array
        for P in Ps:  # manually calculate the mean covariance of each time step across samples
            P_mean += P
        P_mean /= self.num_samples

        # square root to get standard deviations
        sd_x0 = np.array([np.sqrt(Pi[0][0]) for Pi in P_mean])
        sd_x1 = np.array([np.sqrt(Pi[1][1]) for Pi in P_mean])

        return sd_x0, sd_x1

    @staticmethod
    def __get_sample_cov(xs):
        """ calculates the Monte Carlo covariance from the difference between the mean state and each sample
        (xk - x_mean) for each sample xk """
        x_mean = np.array([np.mean(xk) for xk in xs.T])  # calculate mean from each time step for each step
        x_diff = np.array([xk - x_mean for xk in xs])  # subtract mean from each xk
        x_cov = np.array([np.sqrt(np.cov(xi)) for xi in x_diff.T])  # calculate standard deviation for each time step
        return x_cov, x_diff

    def grow_hair(self):
        """ Builds the hairline plot by generating synthetic data, calculating hairlines from each MC sample, and
        assembling a plot for each state variable """
        samples = self.__sample(self.ekf.x, self.ekf.P)  # generate MC samples
        self.meas = self.__gen_measurements(t_eval=self.t, x0=np.ravel(self.ekf.x))  # generate synthetic observations

        # initialize some storage arrays
        x0s = []
        x1s = []
        Ps = []

        ct = 0
        for sample in samples:  # each loop evaluates everything for one sample
            ct += 1
            print("Generating sample", ct)

            self.ekf = deepcopy(self.ekf_init)  # reset self.ekf to the original initialized filter
            # set the original state value of the filter to the sample value
            self.ekf.x = np.array(sample).reshape((self.ekf.dim_x, 1))

            # generate separate measurements for each sample?
            # self.meas = self.gen_measurements(t_eval=self.t, x0=sample)

            # run the filter, updating with the synthetic data
            t_f, x0_f, x1_f, P_f = self.__run_filter(self.ekf, self.t, self.update_freq, self.prop_steps, self.meas)

            Ps.append(P_f)
            x0s.append(x0_f)
            x1s.append(x1_f)

        # get filter standard deviations
        Ps = np.array(Ps)
        sd_x0, sd_x1 = self.__get_filter_cov(Ps)

        # get sample standard deviations
        cov_x0, diff_x0 = self.__get_sample_cov(np.array(x0s))
        cov_x1, diff_x1 = self.__get_sample_cov(np.array(x1s))

        # generate plots
        plt.figure(1)
        plt.title("Population of Species 1")
        plt.xlim([self.t[0], self.t[-1]])
        plt.plot(t_f, 3 * cov_x0, t_f, -3 * cov_x0, color="red", linewidth=2)
        plt.plot(t_f, 3 * sd_x0, t_f, -3 * sd_x0, color="blue", linewidth=2)
        for x0i in diff_x0:
            plt.plot(t_f, x0i, color="green", linewidth=0.25)

        plt.figure(2)
        plt.title("Population of Species 2")
        plt.xlim([self.t[0], self.t[-1]])
        plt.plot(t_f, 3 * cov_x1, t_f, -3 * cov_x1, color="red", linewidth=2)
        plt.plot(t_f, 3 * sd_x1, t_f, -3 * sd_x1, color="blue", linewidth=2)
        for x1i in diff_x1:
            plt.plot(t_f, x1i, color="green", linewidth=0.25)

    @staticmethod
    def show():
        """ shows figures generated in grow_hair() """
        plt.show()
