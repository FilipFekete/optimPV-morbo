"""Module containing classes and functions to visualize the exploration of the parameter space"""

######### Package Imports #########################################################################
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations

from optimpv.axBOtorch.axUtils import get_df_from_ax

######### Function Definitions ####################################################################
def plot_density_exploration(params, optimizer = None, best_parameters = None, params_orig = None, optimizer_type = 'ax', **kwargs):
    """Generate density plots to visualize the exploration of parameter space.

    Parameters
    ----------
    params : list of FitParam() objects
        List of parameters to explore.
    optimizer : object, optional
        Optimizer object, by default None.
    best_parameters : dict, optional
        Dictionary of the best parameters, by default None.
    params_orig : dict, optional
        Dictionary of the original parameters, by default None.
    optimizer_type : str, optional
        Type of optimizer used, by default 'ax'.
    **kwargs : dict
        Additional keyword arguments.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object containing the plot.
    matplotlib.axes._subplots.AxesSubplot
        The axes object containing the plot.

    Raises
    ------
    ValueError
        If the optimizer type is not supported.
    """    

    fig_size = kwargs.get('fig_size', (15, 15))
    levels = kwargs.get('levels', 100)
    
    if optimizer_type == 'ax':
        df = get_df_from_ax(params, optimizer)
    elif optimizer_type == 'pymoo':
        resall = optimizer.all_evaluations
        dum_dic = {}
        for key in resall[0]['params'].keys():
            dum_dic[key] = []
        # for key in resall[0]['results'].keys():
        #     dum_dic[key] = []

        for i in range(len(resall)):
            for key in resall[i]['params'].keys():
                dum_dic[key].append(resall[i]['params'][key])
            # for key in resall[i]['results'].keys():
            #     dum_dic[key].append(resall[i]['results'][key])
        df = pd.DataFrame(dum_dic)
    else:
        raise ValueError('This optimizer type is not supported')
    
        
    names = []
    display_names = []
    log_scale = []
    axis_limits = []
    for p in params:
        if p.type != 'fixed':
            names.append(p.name)
            display_names.append(p.display_name + ' [' + p.unit + ']')
            log_scale.append(p.axis_type == 'log')
            axis_limits.append(p.bounds)


    # Get all combinations of names
    comb = list(combinations(names, 2))

    # Determine the grid size
    n = len(names)
    fig, axes = plt.subplots(n, n, figsize=fig_size)

    # Plot each combination in the grid
    for i, xx in enumerate(names):
        for j, yy in enumerate(names):
            xval = np.nan
            yval = np.nan
            if params_orig is not None:
                xval = params_orig[xx]
                yval = params_orig[yy]

            ax = axes[i, j]
            if i == j:
                # kde plot on the diagonal
                if not np.issubdtype(df[yy].dtype, np.number) :
                    sns.histplot(x=yy, data=df, ax=ax, color="#03051A", log_scale=log_scale[names.index(xx)])
                else:
                    try:
                        sns.kdeplot(x=yy, data=df, ax=ax, fill=True, thresh=0, levels=levels, cmap="rocket", color="#03051A", log_scale=log_scale[names.index(xx)])
                    except:
                        # hystogram if kdeplot fails
                        sns.histplot(x=yy, data=df, ax=ax, color="#03051A", log_scale=log_scale[names.index(xx)])

                if params_orig is not None:
                    ax.axvline(x=yval, color='yellow', linestyle='-')
                if best_parameters is not None:
                    ax.axvline(x=best_parameters[yy], color='r', linestyle='--')
                # put point at the best value top of the axis
            

                if log_scale[names.index(yy)]:
                    ax.set_xscale('log')
                    ax.set_xlim(axis_limits[names.index(yy)])
                else:
                    ax.set_xlim(axis_limits[names.index(yy)])
                
                # put x label on the top
                # except for the last one
                if i < n - 1:
                    ax.xaxis.set_label_position('top')
                    ax.xaxis.tick_top()

            elif i > j:
                kind = 'kde'
                # check for the type of yy and xx in df if they are categorical or not
                if not np.issubdtype(df[yy].dtype, np.number) or not np.issubdtype(df[xx].dtype, np.number):
                    kind = 'scatter'
                if kind == 'scatter':
                    sns.scatterplot(x=yy, y=xx, data=df, ax=ax, color="#03051A")
                    # ax.set_xscale('log')
                    # ax.set_yscale('log')
                else:
                    try:
                        sns.kdeplot(x=yy, y=xx, data=df, ax=ax, fill=True, thresh=0, levels=levels, cmap="rocket", color="#03051A", log_scale=(log_scale[names.index(yy)], log_scale[names.index(xx)]))
                    except Exception as e:
                        print(f"Error in kdeplot: {e}")
                        sns.scatterplot(x=yy, y=xx, data=df, ax=ax, color="#03051A")


                # Plot as line over the full axis
                if params_orig is not None:
                    ax.axhline(y=params_orig[xx], color='yellow', linestyle='-')
                    ax.axvline(x=params_orig[yy], color='yellow', linestyle='-')
                if best_parameters is not None:
                    ax.axhline(y=best_parameters[xx], color='r', linestyle='--')
                    ax.axvline(x=best_parameters[yy], color='r', linestyle='--')
                
                ax.set_xlim(axis_limits[names.index(yy)])
                ax.set_ylim(axis_limits[names.index(xx)])
            else:
                ax.set_visible(False)

            if j > 0:
                if i != j:
                    ax.set_yticklabels([])
                    ax.set_yticklabels([],minor=True)
                    # remove the y axis label
                    ax.set_ylabel('')
            if i < n - 1:
                ax.set_xticklabels([])
                ax.set_xticklabels([],minor=True)
                # remove the x axis label
                ax.set_xlabel('')

            if i == n - 1:
                ax.set_xlabel(display_names[j])
                # for p in params:
                #     if p.name == yy:
                #         ax.set_xlabel(p.display_name + ' [' + p.unit + ']')
                ax.tick_params(axis='x', rotation=45, which='both')
            if j == 0:
                ax.set_ylabel(display_names[i])
                # for p in params:
                #     if p.name == xx:
                #         ax.set_ylabel(p.display_name + ' [' + p.unit + ']')
            if i == j:
                ax.set_title(display_names[i])
                # ax.set_title(params[i].display_name + ' [' +params[i].unit+']')
                ax.set_ylabel('Density')
                ax.yaxis.set_label_position('right')
                ax.yaxis.tick_right()
                ax.yaxis.set_tick_params(which='both', direction='in', left=True, right=True)
                ax.tick_params(axis='y', labelleft=False, labelright=True)

    #custom legend 
    legend_elements = []
    if params_orig is not None:
        legend_elements.append(plt.Line2D([0], [0], color='yellow', label='Original parameters', linestyle='-'))
    if best_parameters is not None:
        legend_elements.append(plt.Line2D([0], [0], color='r', label='Best parameters', linestyle='--'))
    if len(legend_elements) > 0:
        fig.legend(handles=legend_elements, loc='center right', bbox_to_anchor=(0.9, 0.5), ncol=1)

    # change spacing between subplots
    plt.tight_layout()
    fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, hspace=0.2, wspace=0.2)
    
    return fig, axes
