"""Module containing classes and functions for posterior analysis of parameters using the ML models from the BO optimization.
This module provides functionality to visualize the posterior distributions of parameters
using various plots, including 1D and 2D posteriors, devil's plots, and density plots."""

######### Package Imports #########################################################################
import copy, itertools, scipy
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from optimpv.general.BaseAgent import BaseAgent
from scipy.stats import gaussian_kde
######### Function Definitions ####################################################################

class LazyPosterior(BaseAgent):



    def __init__(self, params, df, outcome_name, best_params = None, is_nat_scale = False, maximize=True, **kwargs):
        """ LazyPosterior class object to visualize the approximate posterior.

        Parameters
        ----------
        params : list of Fitparam() objects
            list of Fitparam() objects defining the parameters to optimize
        df : pd.DataFrame
            DataFrame containing the data used for training the surrogate model
        outcome_name : str
            Name of the outcome variable in the DataFrame
        model : BotorchModel, optional
            Surrogate model used for the approximate posterior, by default None
        best_params : dict, optional
            Dictionary of best parameter values, by default None
        is_nat_scale : bool, optional
            Indicates if the parameters are in natural scale, by default False
        maximize : bool, optional
            Indicates if the outcome is to be maximized, by default True
        """          
        
        self.params = params
        params_names = [param.name for param in params if not param.type == 'fixed']
        params_names_to_index = {name: idx for idx, name in enumerate(params_names)}
        self.params_names_to_index = params_names_to_index
        self.outcome_name = outcome_name
        self.kwargs = kwargs
        self.name_free_params = [p.name for p in self.params if not p.type == 'fixed']
        self.num_free_params = len(self.name_free_params)
        self.param_by_name = {p.name: p for p in self.params}
        if best_params is None:
            best_idx = df[outcome_name].idxmax()
            best_params = df.loc[best_idx][params_names].to_dict()

        # get the dataframe in the rescaled space and keep the true dataframe, df and best_params are always rescaled and will be used for training the surrogate model
        if is_nat_scale:
            self.true_df = df
            self.df = self.descale_dataframe(copy.deepcopy(df), params)
            self.true_best_params = best_params
            self.best_params = self.params_descale(best_params, params)
        else:
            self.true_df = self.rescale_dataframe(copy.deepcopy(df), params)
            self.df = df
            self.true_best_params = self.params_rescale(best_params, params)
            self.best_params = best_params
        # add the fixed parameters to the true_best_params and true_df
        for p in self.params:
            if p.type == 'fixed':
                self.true_best_params[p.name] = p.value
                self.true_df[p.name] = p.value*len(self.true_df)

        self.maximize = maximize
        self.ascending = not maximize

        
        
    def plot_lazyposterior_2D_kde(self, x_param_name, y_param_name, ax = None, **kwargs):
        """ Plot the 2D posterior KDE for two parameters.

        Parameters
        ----------
        x_param_name : str
            Name of the first parameter
        y_param_name : str
            Name of the second parameter
        ax : matplotlib.axes.Axes, optional
            Matplotlib Axes object to plot on. If None, a new figure and axes are created.
        **kwargs : dict
            Additional keyword arguments passed to seaborn.kdeplot:
            - levels: int, optional, default=10
                Number of contour levels to draw.
            - cmap: str or Colormap, optional, default='viridis'
                Colormap to use for the plot.
            - alpha: float, optional, default=0.7
                Transparency level for the filled contours.
            - title: str, optional
                Title of the plot.
            - xlabel: str, optional
                Label for the x-axis.

        """          
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(8,6))
        
        for p in self.params:
            if p.name == x_param_name:
                x_param = p.full_name
                xlim = p.bounds
                xlog = p.log_scale or p.force_log
            if p.name == y_param_name:
                y_param = p.full_name
                ylim = p.bounds
                ylog = p.log_scale or p.force_log

        x = self.true_df[x_param_name].values
        y = self.true_df[y_param_name].values
        sns.kdeplot(
            x=x,
            y=y,
            fill=True,
            ax=ax,
            levels=kwargs.get('levels', 10),
            cmap=kwargs.get('cmap', 'viridis'),
            alpha=kwargs.get('alpha', 0.7),
            log_scale=(xlog, ylog)
            
        )
        ax.set_xlabel(kwargs.get('xlabel', x_param))
        ax.set_ylabel(kwargs.get('ylabel', y_param))

        if xlog:
            ax.set_xscale(kwargs.get('xscale', 'log'))
        if ylog:
            ax.set_yscale(kwargs.get('yscale', 'log'))
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)

        ax.set_title(kwargs.get('title', f'2D KDE Posterior of {x_param_name} and {y_param_name}'))
        plt.tight_layout()

        return ax

    def plot_lazyposterior_1D_kde(self, param_name, ax = None, **kwargs):
        """ Plot the 1D posterior KDE for a parameter.

        Parameters
        ----------
        param_name : str
            Name of the parameter
        ax : matplotlib.axes.Axes, optional
            Matplotlib Axes object to plot on. If None, a new figure and axes are created.
        **kwargs : dict
            Additional keyword arguments passed to seaborn.kdeplot:
            - color: str, optional, default='blue'
                Color of the KDE line.
            - fill: bool, optional, default=True
                Whether to fill the area under the KDE curve.
            - alpha: float, optional, default=0.5
                Transparency level for the filled area.
            - title: str, optional
                Title of the plot.
            - xlabel: str, optional
                Label for the x-axis.

        """          
  
        if ax is None:
            fig, ax = plt.subplots(figsize=(8,6))
        
        for p in self.params:
            if p.name == param_name:
                param_full_name = p.full_name
                param_bounds = p.bounds
                param_log = p.log_scale or p.force_log

        # darkest viridis color
        x = self.true_df[param_name].values
        sns.kdeplot(
            x=x,
            ax=ax,
            color=kwargs.get('color', 'C0'),
            fill=kwargs.get('fill', False),
            alpha=kwargs.get('alpha', 0.5),
            log_scale=param_log
        )
        ax.set_xlabel(kwargs.get('xlabel', param_full_name))
        ax.set_ylabel(kwargs.get('ylabel', 'Density'))
        if param_log:
            ax.set_xscale(kwargs.get('xscale', 'log'))
        if param_bounds is not None:
            ax.set_xlim(param_bounds)
        ax.set_title(kwargs.get('title', f'1D KDE Posterior of {param_name}'))
        

        # add all metrics lines if requested
        if kwargs.get('show_metrics', True):
            mean, median, hdi_lower, hdi_upper, max_value = self.compute_metrics(param_name, 'all', credibility_level=0.95)
            ax.axvline(mean, color='C4', linestyle=':', label='Mean')
            ax.axvline(median, color='C2', linestyle='-.', label='Median')
            ax.hlines(y=0.05, xmin=hdi_lower, xmax=hdi_upper, color='k', linestyle='-', linewidth=2, label='95% HDI')
            ax.axvline(max_value, color='C3', linestyle='--', label='Max Value')
        
        if kwargs.get('show_top_n', None) is not None:
            top_n = kwargs.get('show_top_n')
            top_n_params_list = self.get_top_n_best_params(num=top_n)
            # get key positions of param_name in self.params_by_name key list
            pos = self.name_free_params.index(param_name)
            # get maximum density value for scaling vertical lines
            kde = self._compute_1D_kde(param_name, log_scale=param_log)
            x_min, x_max = self.true_df[param_name].min(), self.true_df[param_name].max()
            if param_log:
                x_points = np.logspace(np.log10(x_min), np.log10(x_max), 1000)
            else:
                x_points = np.linspace(x_min, x_max, 1000)
            kde_values = kde(x_points)
            max_density = np.max(kde_values)
            for i, params in enumerate(top_n_params_list):
                p_val = params[pos].value
                # heith of the line need to go from the bottom axis to 0.05*max_density
                ax.vlines(p_val, ymin=0, ymax=0.05*max_density, color='gray', linestyle='-', alpha=0.7,linewidth=1, label=f'Top {top_n}' if i==0 else None)
                
        
        if kwargs.get('show_legend', True):
            ax.legend(loc='best', ncol=2)
        plt.tight_layout()

        return ax
    
    def plot_all_lazyposterior(self, fig=None, axes=None, **kwargs):
        """ Plot the 1D and 2D posterior KDEs for all free parameters.

        Parameters
        ----------
        fig : matplotlib.figure.Figure, optional
            Matplotlib Figure object to plot on. If None, a new figure is created.
        axes : numpy.ndarray of matplotlib.axes.Axes, optional
            Array of Matplotlib Axes objects to plot on. If None, new axes are created.
        **kwargs : dict
            Additional keyword arguments passed to seaborn.kdeplot.
        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object containing the plots.
        axes : numpy.ndarray of matplotlib.axes.Axes
            The axes object containing the plots.   
        """


        num_params = self.num_free_params
        fig, axes = plt.subplots(num_params, num_params, figsize=kwargs.get('figsize', (4*num_params, 4*num_params)))
        param_names = [p.name for p in self.params if not p.type == 'fixed']

        # build quick lookup from name -> param object for bounds/display
        param_by_name = {p.name: p for p in self.params}

        for row_idx, col_idx in itertools.product(range(num_params), range(num_params)):
            x_name = param_names[col_idx]  # column -> x-axis param
            y_name = param_names[row_idx]  # row -> y-axis param
            ax = axes[row_idx, col_idx]

            # Diagonal: 1D KDE
            if row_idx == col_idx:

                ax = sns.kdeplot(
                    x=self.true_df[x_name].values,
                    ax=ax,
                    color=kwargs.get('color', 'C0'),
                    fill=kwargs.get('fill', False),
                    alpha=kwargs.get('alpha', 0.5),
                    log_scale=param_by_name[x_name].log_scale or param_by_name[x_name].force_log
                )
                ax.set_xlim(param_by_name[x_name].bounds[0], param_by_name[x_name].bounds[1])
                p = param_by_name[x_name]
                # y-labels: left column shows ylabel, other columns show right ticks/empty
                if col_idx == 0:
                    ax.set_ylabel(kwargs.get('ylabel_diag','Density'))
                else:
                    ax.yaxis.tick_right()
                    ax.yaxis.set_label_position("right")
                    ax.set_ylabel('')

                # x-label only on bottom row
                if row_idx == num_params - 1:
                    ax.set_xlabel(f"{p.full_name}")
                else:
                    ax.set_xticklabels([])
            # Off-diagonal: 2D KDE
            elif row_idx > col_idx:
                xlog = param_by_name[x_name].log_scale or param_by_name[x_name].force_log
                ylog = param_by_name[y_name].log_scale or param_by_name[y_name].force_log

                sns.kdeplot(
                    x=self.true_df[x_name].values,
                    y=self.true_df[y_name].values,
                    fill=True,
                    ax=ax,
                    levels=kwargs.get('levels', 10),
                    cmap=kwargs.get('cmap', 'viridis'),
                    alpha=kwargs.get('alpha', 0.7),
                    log_scale=(xlog, ylog)
                )
                 # set axis bounds from parameter definitions
                px = param_by_name[x_name]
                py = param_by_name[y_name]
                ax.set_xlim(px.bounds[0], px.bounds[1])
                ax.set_ylim(py.bounds[0], py.bounds[1])

                # log scales if requested
                if xlog:
                    ax.set_xscale(kwargs.get('xscale', 'log'))
                if ylog:
                    ax.set_yscale(kwargs.get('yscale', 'log'))

                # x-label only for bottom row (same x label for that column)
                if row_idx == num_params - 1:
                    ax.set_xlabel(f"{px.display_name} [{px.unit}]")
                else:
                    ax.set_xticklabels([])

                # y-label only for first column (same y label for that row)
                if col_idx == 0:
                    ax.set_ylabel(f"{py.display_name} [{py.unit}]")
                else:
                    ax.set_yticklabels([])
            else:
                ax.axis('off')

        plt.tight_layout()
        return fig, axes
    

    def _compute_1D_kde(self, param_name, log_scale=False, **kwargs):
        """ Compute the 1D KDE for a parameter.

        Parameters
        ----------
        param_name : str
            Name of the parameter
        log_scale : bool, optional
            Whether to compute the KDE on a log scale, by default False

        Returns
        -------
        kde : scipy.stats.gaussian_kde
            The computed KDE object.

        Raises
        ------
        ValueError
            if any values are <= 0 when log_scale is True
        """        
        x = self.true_df[param_name].values

        if log_scale:
            if np.any(x <= 0):
                raise ValueError("All values must be > 0 for log-scale KDE.")
            y = np.log10(x)
            kde_log = gaussian_kde(y, **kwargs)

            # Wrapper to transform back to original scale
            def kde(x_orig):
                x_log = np.log10(x_orig)
                return kde_log(x_log)  # Adjust for change of variables
            return kde
        else:
            kde = gaussian_kde(x, **kwargs)
            return kde
        
    def compute_high_density_interval(self, param_name, credibility_level=0.95, num_points=1000):
        """ Compute the highest density interval (HDI) for a parameter.

        Parameters
        ----------
        param_name : str
            Name of the parameter
        credibility_level : float, optional
            Credibility level for the HDI, by default 0.95
        num_points : int, optional
            Number of points to evaluate the KDE, by default 1000

        Returns
        -------
        hdi_lower : float
            Lower bound of the HDI
        hdi_upper : float
            Upper bound of the HDI
        """        
        # Get parameter object to check for log scale
        param = next(p for p in self.params if p.name == param_name)
        log_scale = param.log_scale or param.force_log

        kde = self._compute_1D_kde(param_name, log_scale=log_scale)

        # Generate points over the parameter range
        x_min, x_max = self.true_df[param_name].min(), self.true_df[param_name].max()
        x_points = np.linspace(x_min, x_max, num_points)
        if log_scale:
            x_points = np.logspace(np.log10(x_min), np.log10(x_max), num_points)

        # Evaluate KDE at these points
        kde_values = kde(x_points)

        # Sort points by density
        sorted_indices = np.argsort(kde_values)[::-1]
        sorted_x = x_points[sorted_indices]
        sorted_kde = kde_values[sorted_indices]

        # Compute cumulative density
        cumulative_density = np.cumsum(sorted_kde)
        cumulative_density /= cumulative_density[-1]  # Normalize to 1

        # Find HDI bounds
        indices_in_hdi = sorted_indices[cumulative_density <= credibility_level]
        hdi_x_values = x_points[indices_in_hdi]

        hdi_lower = np.min(hdi_x_values)
        hdi_upper = np.max(hdi_x_values)

        return hdi_lower, hdi_upper
    
    def compute_metrics(self, param_name, metric_name, credibility_level=0.95, num_points=100):
        """ Compute the mean, median, and HDI for a parameter.

        Parameters
        ----------
        param_name : str
            Name of the parameter
        metric_name : str
            Metric to compute: 'mean', 'median', 'hdi', 'max', or 'all'
        credibility_level : float, optional
            Credibility level for the HDI, by default 0.95
        num_points : int, optional
            Number of points to evaluate the KDE, by default 100
        Returns
        -------
        mean : float
            Mean of the parameter distribution
        median : float
            Median of the parameter distribution
        hdi_lower : float
            Lower bound of the HDI
        hdi_upper : float
            Upper bound of the HDI
        max_value : float
            value at the maximum of the parameter distribution
        """        
        # Get parameter object to check for log scale
        param = next(p for p in self.params if p.name == param_name)
        log_scale = param.log_scale or param.force_log

        kde = self._compute_1D_kde(param_name, log_scale=log_scale)

        # Generate points over the parameter range
        bounds = self.params[self.params_names_to_index[param_name]].bounds
        x_points = np.linspace(bounds[0], bounds[1], num_points)
        if log_scale:
            x_points = np.logspace(np.log10(bounds[0]), np.log10(bounds[1]), num_points)
        # Evaluate KDE at these points
        kde_values = kde(x_points)
        
        if metric_name == 'mean':
            if log_scale:
                log_x_points = np.log10(x_points)
                mean_log = np.sum(log_x_points * kde_values) / np.sum(kde_values)
                mean = 10**mean_log
            else:
                mean = np.sum(x_points * kde_values) / np.sum(kde_values)
            return mean
        elif metric_name == 'median':
            cdf = np.cumsum(kde_values)
            cdf /= cdf[-1]  # Normalize CDF to 1
            median = np.interp(0.5, cdf, x_points)
            return median
        elif metric_name == 'hdi':
            if credibility_level <= 0 or credibility_level >= 1:
                raise ValueError("credibility_level must be between 0 and 1.")
            hdi_lower, hdi_upper = self.compute_high_density_interval(param_name, credibility_level, num_points)
            return hdi_lower, hdi_upper
        elif metric_name == 'max':
            max_idx = np.argmax(kde_values)
            max_value = x_points[max_idx]
            return max_value
        elif metric_name == 'all':
            mean = np.sum(x_points * kde_values) / np.sum(kde_values)
            cdf = np.cumsum(kde_values)
            cdf /= cdf[-1]  # Normalize CDF to 1
            median = median = np.interp(0.5, cdf, x_points)
            hdi_lower, hdi_upper = self.compute_high_density_interval(param_name, credibility_level, num_points)
            max_idx = np.argmax(kde_values)
            max_value = x_points[max_idx]
            return mean, median, hdi_lower, hdi_upper, max_value
        else:
            raise ValueError("metric_name must be 'mean', 'median', 'hdi', 'max', or 'all'.")
        

    def get_median_params(self):
        """ Get the median parameter values for all free parameters."""

        median_params = copy.deepcopy(self.params)

        for p in median_params:
            median_value = self.compute_metrics(p.name, 'median')
            p.value = median_value
        
        return median_params
    
    def get_max_params(self):
        """ Get the parameter values at the maximum of the posterior for all free parameters."""

        max_params = copy.deepcopy(self.params)

        for p in max_params:
            max_value = self.compute_metrics(p.name, 'max')
            p.value = max_value
        
        return max_params
    
    def get_mean_params(self):
        """ Get the mean parameter values for all free parameters."""

        mean_params = copy.deepcopy(self.params)

        for p in mean_params:
            mean_value = self.compute_metrics(p.name, 'mean')
            p.value = mean_value
        
        return mean_params
    
    def get_top_n_best_params(self, num = 10):
        """ Make x params list with the top x best parameters from the true dataframe."""
        x_best_params_list = []
        for i in range(num):
            x_best_params = copy.deepcopy(self.params)
            sorted_df = self.true_df.sort_values(by=self.outcome_name, ascending=self.ascending).reset_index()
            dum_params = copy.deepcopy(self.params)
            for p in dum_params:
                p.value = sorted_df.loc[i][p.name]
            x_best_params_list.append(dum_params)
        return x_best_params_list
    
    def get_top_n_metrics_params(self, metric_name, num = 10):
        """ Get the parameter values computed from the top n best parameters from the true dataframe.

        Parameters
        ----------
        metric_name : str
            Metric to compute from the top n best parameters: 'mean', 'median', 'max', or 'min'
        num : int, optional
            Number of top best parameters to consider, by default 10

        Returns
        -------
        list
            List of parameter objects with values computed from the specified metric of the top n best parameters.

        Raises
        ------
        ValueError
            if metric_name is not 'mean', 'median', 'max', or 'min'
        """        
       
        x_best_params_list = self.get_top_n_best_params(num = num)
        dum_params = copy.deepcopy(self.params)
        params_list_dic = {}
        for p in self.params:
            
            dum_params_values = []
            for params in x_best_params_list:
                for param in params:
                    if param.name == p.name:
                        dum_params_values.append(param.value)
            params_list_dic[p.name] = dum_params_values
        
        for p in dum_params:
            if metric_name == 'mean':
                metric_value = np.mean(params_list_dic[p.name])
            elif metric_name == 'median':
                metric_value = np.median(params_list_dic[p.name])
            elif metric_name == 'max':
                metric_value = np.max(params_list_dic[p.name])
            elif metric_name == 'min':
                metric_value = np.min(params_list_dic[p.name])
            else:
                raise ValueError("metric_name must be 'mean', 'median', 'max', or 'min'.")
            p.value = metric_value
        return dum_params
       
    
    def get_best_params(self):
        """ Get the best parameter values from the true dataframe."""
        best_params = copy.deepcopy(self.params)
        for p in best_params:
            p.value = self.true_best_params[p.name]
        return best_params