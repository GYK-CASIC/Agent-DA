#!/usr/bin/env python3
"""
Logger during the training process, similar to the SummaryWriter function, but SummaryWriter relies on TensorBoard and a browser for visualization.
This tool uses matplotlib to store static local images, making it convenient to quickly check training results on the server.
"""
import os

import numpy as np
import matplotlib.pyplot as plt


class iSummaryWriter(object):

    def __init__(self, log_path: str, log_name: str, params=[], extention='.png', max_columns=2,
                 log_title=None, figsize=None):
        """
        Initialization function, create a log class.

        Args:
            log_path (str): Log storage folder
            log_name (str): Log file name
            params (list): List of parameter names to record, e.g. -> ["loss", "reward", ...]
            extension (str): Image storage format
            max_columns (int): Number of images per row, default is 2 images (2 variables) per row.
        """
        self.log_path = log_path
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        self.log_name = log_name
        self.extention = extention
        self.max_param_index = -1
        self.max_columns_threshold = max_columns
        self.figsize = figsize
        self.params_dict = self.create_params_dict(params)
        self.log_title = log_title
        self.init_plt()
        self.update_ax_list()

    def init_plt(self) -> None:
        plt.style.use('seaborn-darkgrid')

    def create_params_dict(self, params: list) -> dict:
        """
        Create a monitoring variable dictionary based on the list of variable names to be recorded.

        Args:
            params (list): List of monitoring variable names

        Returns:
            dict: Monitoring variable name dictionary -> {
                'loss': {'values': [0.44, 0.32, ...], 'epochs': [10, 20, ...], 'index': 0},
                'reward': {'values': [10.2, 13.2, ...], 'epochs': [10, 20, ...], 'index': 1},
                ...
            }
        """
        params_dict = {}
        for i, param in enumerate(params):
            params_dict[param] = {'values': [], 'epochs': [], 'index': i}
            self.max_param_index = i
        return params_dict

    def update_ax_list(self) -> None:
        """
        Allocate a plot area for each variable according to the current monitoring variable dictionary.
        """
        # * Recalculate the plot index corresponding to each variable
        params_num = self.max_param_index + 1
        if params_num <= 0:
            return

        self.max_columns = params_num if params_num < self.max_columns_threshold else self.max_columns_threshold
        max_rows = (params_num - 1) // self.max_columns + 1   # * Maximum number of rows for all variables
        figsize = self.figsize if self.figsize else (self.max_columns * 6, max_rows * 3)    # Calculate the figsize of the entire figure based on the number of plots
        self.fig, self.axes = plt.subplots(max_rows, self.max_columns, figsize=figsize)

        # * If there is only one row but more than one plot, manually reshape to (1, n) form
        if params_num > 1 and len(self.axes.shape) == 1:
            self.axes = np.expand_dims(self.axes, axis=0)

        # * Reset log title
        log_title = self.log_title if self.log_title else '[Training Log] {}'.format(
            self.log_name)
        self.fig.suptitle(log_title, fontsize=15)

    def add_scalar(self, param: str, value: float, epoch: int) -> None:
        """
        Add a new variable value record.

        Args:
            param (str): Variable name, e.g. -> 'loss'
            value (float): Value at this time.
            epoch (int): Epoch number at this time.
        """
        # * If this parameter is added for the first time, add it to the monitoring variable dictionary
        if param not in self.params_dict:
            self.max_param_index += 1
            self.params_dict[param] = {'values': [],
                                       'epochs': [], 'index': self.max_param_index}
            self.update_ax_list()

        self.params_dict[param]['values'].append(value)
        self.params_dict[param]['epochs'].append(epoch)

    def record(self, dpi=200) -> None:
        """
        Call this interface to record the current state of all monitored variables and save the result to a local file.
        """
        for param, param_elements in self.params_dict.items():
            param_index = param_elements["index"]
            param_row, param_column = param_index // self.max_columns, param_index % self.max_columns
            ax = self.axes[param_row, param_column] if self.max_param_index > 0 else self.axes
            # ax.set_title(param)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(param)
            ax.plot(self.params_dict[param]['epochs'],
                    self.params_dict[param]['values'],
                    color='darkorange')

        plt.savefig(os.path.join(self.log_path,
                    self.log_name + self.extention), dpi=dpi)


if __name__ == '__main__':
    import random
    import time

    n_epochs = 10
    log_path, log_name = './', 'test'
    writer = iSummaryWriter(log_path=log_path, log_name=log_name)
    for i in range(n_epochs):
        loss, reward = 100 - random.random() * i, random.random() * i
        writer.add_scalar('loss', loss, i)
        writer.add_scalar('reward', reward, i)
        writer.add_scalar('random', reward, i)
        writer.record()
        print("Log has been saved at: {}".format(
            os.path.join(log_path, log_name)))
        time.sleep(3)
