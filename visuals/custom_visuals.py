import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os


class Visuals(object):
    """
    Class containing routines to save system snapshots in terms of images.
    """

    def __init__(self, columns=['population', 'N_settlements', 'Malthus'],
                 shape=None, t_max=None, location='pics'):
        """
        Init function of the Visuals class. Saves parameters, creates figure.

        Parameters
        ----------
        columns : list of strings
            list of column names for the subplots that are not the map
        shape : tuple or its
            shape of the map
        t_max : int
            max number of timesteps to set xlim in trajectory plots.
            not implemented yet.
        location : string
            path where the snapshot images will be saved
        """

        # make saving location for plots, in case it does not exist
        self.location = location.rstrip('/')
        if not os.path.isdir(self.location):
            os.makedirs(self.location)
        print(self.location)

        # image/frame counter for naming
        self.i_image = 0
        self.shape = shape
        # shapes of subplot grid
        self.ylen = len(columns)
        self.xlen = self.ylen + 1
        self.columns = columns
        # list to save trajectory of ad hoc calculated macro quantities.
        self.trajectory = []

        # create figure with some small plots on the left
        # and a big plot on the right.
        self.figure = plt.figure(figsize=(11.7, 8.3))
        self.axes = []
        for c, column in enumerate(self.columns):
            self.axes.append(plt.subplot2grid((self.ylen, self.xlen),
                                              (c, 0)))
            self.axes[-1].set_title(column)
        self.axes.append(plt.subplot2grid((self.ylen, self.xlen),
                                          (0, 1), rowspan=self.xlen,
                                          colspan=self.xlen))

    def update_plots(self, population, real_income, ag_income, es_income,
                     trade_income, adjacency, settlement_positions):
        """
        Creates a snapshot plot from the parameters passed.

        Parameters
        ----------
        population: list of floats
            The population of all the existing settlements,
        real_income: list of floats
            The real income of all existing settlements,
        ag_income: list of floats
            The income from agriculture per capita for all settlements,
        es_income: list of floats
            The income from ecosystem services per capita for all settlements,
        trade_income: list of floats
            The income from trade per capita for all settlements,
        adjacency: ndarray
            The adjacency matrix of the trade network between settlements,
        settlement_positions: list of lists of floats
            The x and y positions of all settlements on the map.

        Returns
        -------
        self.figure: object
            the updated matplotlib figure object.

        """

        # increment frame counter
        self.i_image += 1
        # append ad hoc macro quantitites to the trajectory
        self.trajectory.append([sum(population), len(population)])

        # plot macro trajectories to subplots
        for c, column in enumerate(self.columns[:2]):
            c_data = [d[c] for d in self.trajectory]
            self.axes[c].clear()
            self.axes[c].plot(c_data)
            self.axes[c].set_title(column)
            self.axes[c].locator_params(nbins=3)

        # clear scatterplot and prepare colors
        # (fraction of ag income in total income)
        axm = self.axes[-2]
        axm.clear()
        cmap = mpl.cm.BrBG
        colors = [cmap(1. - float(ag_income[i]) / float(r_income))
                  if float(r_income) > 0 else (0, 0, 0, 1)
                  for i, r_income in enumerate(real_income)]

        # make scatterplot
        try:
            axm.scatter(population, real_income, c=colors)
        except ValueError:
            print('printing error!!!')
            print(population, real_income, colors)

        # adjust axis settings
        axm.locator_params(nbins=3)
        axm.set_xlim([0., 50000.])
        axm.set_ylim([0., 2.])
        axm.set_title(self.columns[-1])

        # clear trade network plot
        ax = self.axes[-1]
        ax.clear()
        ax.set_title('Trade-Network')
        x = settlement_positions[0] + 0.5
        y = settlement_positions[1] + 0.5

        # plot settlements with color acccording to trade income
        cmap = mpl.cm.Blues
        t_max = max(trade_income)
        colors = [cmap(t / t_max) for t in trade_income]
        try:
            ax.scatter(y, x, c=colors, zorder=2)
        except ValueError:
            print(t_max, ' t_max might have weird values?')

        # plot trade network
        generator = (i for i, x in np.ndenumerate(adjacency) if
                     adjacency[i] == 1)
        for i, j in generator:
            ax.plot([y[i], y[j]], [x[i], x[j]], color="k", linewidth=0.5,
                    zorder=1)

        # set shape of map and save.
        if self.shape is not None:
            ax.set_xlim([0, self.shape[1]])
            ax.set_ylim([self.shape[0], 0])

        return self.figure