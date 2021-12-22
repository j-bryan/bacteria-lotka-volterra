import matplotlib.pyplot as plt
import numpy as np


class StreamPlot:

    """ Creates a plot object to easier work with and fine-tune the look of a matplotlib.pyplot streamplot

    ----------------
    -- Parameters --
    ----------------
        :param fun : (function) differential equation with two-dimensional output
        :param r : (function) growth rates; two-dimensional output
        :param k : (float) carrying capacity
        :param plot_gradient : (boolean; optional - default: True) include gradient background
        :param plot_trajectories : (boolean; optional - default: False) include data trajectories
        :param plot_colorbar : (boolean; optional - default: False) include gradient colorbar
        :param plot_rates : (boolean; optional - default: False) include growth rates subplot

    ----------------
    -- Attributes --
    ----------------
        fun : function (P0, P1, r0, r1, k)
            differential equation used to create phase diagrams
            returns the rate of change of each species' population
        r : function (x)
            intrinsic growth rate of each population for a percentage P0 / (P0 + P1)
        k : float
            carrying capacity
        plot_gradient : boolean
            activates gradient background plot of population rate of change
        plot_trajectories : boolean
            activates plotting of real data on phase diagram
        plot_colorbar : boolean
            shows colorbar describing gradient background
        plot_rates : boolean
            adds a subplot of the population intrinsic growth rates
        x_min : float
            x axis minimum
        x_max : float
            x axis maximum
        x_int : float
            number of subintervals in x
        y_min : float
            y axis minimum
        y_max : float
            y axis maximum
        y_int : float
            number of subintervals in y
        axis_lims : array of six (6) floats
            used to succinctly update all x and y min/max/interval values; values must be passed in in form
            [x_min, x_max, x_int, y_min, y_max, y_int]
            to change individual axis bounds, use the individual attribute for the bound in question
        filepath : string
            file path to save figure
        linewidth : float
            line width of liens in streamplot
        density : float
            density of lines in streamplot
        line_color : string
            color of streamplot lines
        colormap : pyplot colormap (matplotplib.pyplot.cm)
            colormap to create background gradient image
        traj_data : 2-dimensional array
                population data to be plotted over the phase diagram
                data must be in form [[A1, B1], [A2, B2], ... ]

    -------------
    -- Example --
    -------------
        # create object
        strm = StreamPlot(fun=dP_dt, r=r, k=10)

        # trajectory data generation using scipy.integrate.solve_ivp
        sol = solve_ivp(fun=dP_dt_wrapper, t_span=[0, 30], y0=[2, 10])
        C1 = sol.y[0]
        D1 = sol.y[1]

        # adjust plot attributes and features
        strm.axis_lims = [0, 12, 1000, 0, 12, 1000]
        strm.traj_data = [[C1, D1]]
        strm.plot_rates = True
        strm.plot_trajectories = True
        strm.plot_colorbar = True

        # show figure
        strm.show()
        # save figure
        strm.save("path/to/save/location.png")
        """

    def __init__(self, fun, r, k, plot_gradient=True, plot_trajectories=False, plot_colorbar=False, plot_rates=False):
        # model
        self.fun = fun
        self.r = r
        self.k = k

        # main plot
        # plot figure and axis objects
        self.fig, self.ax = plt.subplots()
        # default plot boundaries and number of subintervals
        self.x_min = 0
        self.x_max = 1
        self.x_int = 100
        self.y_min = 0
        self.y_max = 1
        self.y_int = 100
        self.axis_lims = None

        # save filepath
        self.filepath = None
        # aesthetic tweaks
        self.linewidth = 0.5
        self.density = 1.5
        self.line_color = "#ff2200"

        # gradient background
        self.plot_gradient = plot_gradient
        self.colormap = plt.cm.copper
        self.grad_img = None

        # trajectories
        self.plot_trajectories = plot_trajectories
        self.traj_data = None
        self.trajectory_colors = ['orange', 'blue', 'green', 'purple']
        self.trajectory_start_points = []
        self.trajectory_end_points = []

        # colorbar
        self.plot_colorbar = plot_colorbar

        # rates subplot
        self.plot_rates = plot_rates
        self.__subcoords_w_colorbar = [0.485, 0.62, 0.25, 0.25]
        self.__subcoords_wo_colorbar = [0.54, 0.62, 0.25, 0.25]

    def gradient(self, U, V):
        speed = np.sqrt(U ** 2 + V ** 2)
        self.grad_img = plt.imshow(speed[::-1], extent=(self.x_min, self.x_max, self.y_min, self.y_max),
                                   cmap=self.colormap)

    def rates(self):
        if self.plot_colorbar:
            sub_ax = self.fig.add_axes(self.__subcoords_w_colorbar)
        else:
            sub_ax = self.fig.add_axes(self.__subcoords_wo_colorbar)
        sub_ax.spines['top'].set_color('white')
        sub_ax.spines['bottom'].set_color('white')
        sub_ax.spines['left'].set_color('white')
        sub_ax.spines['right'].set_color('white')
        sub_ax.tick_params(colors='white')
        x = np.arange(0, 1, 0.01)
        rc, rd = self.r(x)
        sub_ax.plot(x, rc, label='r_C')
        sub_ax.plot(x, rd, label='r_D')
        sub_ax.legend()
        plt.xlim([0, 1])
        plt.ylim([0, 1])

    def colorbar(self):
        plt.colorbar(self.grad_img)

    def trajectories(self):
        ct = 0
        for set in self.traj_data:
            plt.plot(set[0], set[1], self.trajectory_colors[ct])
            self.trajectory_start_points.append((set[0][0], set[1][0]))
            self.trajectory_end_points.append((set[0][-1], set[1][-1]))
            ct += 1

        ct = 0
        for line in zip(self.trajectory_start_points, self.trajectory_end_points):
            line = np.array(line)
            plt.scatter(line[0, 0], line[0, 1], marker='o', color=self.trajectory_colors[ct], zorder=2)
            plt.scatter(line[1, 0], line[1, 1], marker='x', color=self.trajectory_colors[ct], zorder=2)
            ct += 1

    def save(self, filepath):
        self.filepath = filepath
        self.show(show_fig=False)
        plt.savefig(filepath)

    def show(self, show_fig=True, filepath=None):
        if self.axis_lims is not None:
            self.x_min = self.axis_lims[0]
            self.x_max = self.axis_lims[1]
            self.x_int = self.axis_lims[2]
            self.y_min = self.axis_lims[3]
            self.y_max = self.axis_lims[4]
            self.y_int = self.axis_lims[5]

        plt.xlim([self.x_min, self.x_max])
        plt.ylim([self.y_min, self.y_max])

        Y, X = np.mgrid[self.x_min:self.x_max:np.complex(0, self.x_int), self.y_min:self.y_max:np.complex(0, self.y_int)]
        x = X / (X + Y)
        rc, rd = self.r(x)
        U, V = self.fun(X, Y, rc, rd, self.k)

        plt.streamplot(X, Y, U, V, linewidth=self.linewidth, color=self.line_color, density=self.density)

        if self.plot_gradient:
            self.gradient(U, V)
        if self.plot_colorbar and self.plot_gradient:
            self.colorbar()
        if self.plot_trajectories and self.traj_data is not None:
            self.trajectories()
        if self.plot_rates:
            self.rates()

        if filepath is not None:
            plt.savefig(filepath)
        if show_fig:
            plt.show()
