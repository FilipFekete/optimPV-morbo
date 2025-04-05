"""Provides general functionality for Agent objects for non ideal diode simulations"""
######### Package Imports #########################################################################

import os, uuid, sys, copy, warnings
import numpy as np
import pandas as pd
from scipy import interpolate, constants

try: 
    import pvlib
    from pvlib.pvsystem import i_from_v
    got_pvlib = True
except:
    got_pvlib = False
    warnings.warn('pvlib not installed, using scipy for diode equation')

from optimpv import *
from optimpv.general.general import calc_metric, loss_function, transform_data
from optimpv.general.BaseAgent import BaseAgent
from optimpv.Diodefits.DiodeModel import *

## Physics constants
q = constants.value(u'elementary charge')
eps_0 = constants.value(u'electric constant')
kb = constants.value(u'Boltzmann constant in eV/K')

######### Agent Definition #######################################################################
class DiodeAgent(BaseAgent):
    """Agent object for non ideal diode simulations
    with the following formula:
    J = Jph - J0*[exp(-(V-J*R_series)/(n*Vt*)) - 1] - (V - J*R_series)/R_shunt
    see optimpv.Diodefits.DiodeModel.py for more details

    Parameters
    ----------
    params : list of Fitparam() objects
        List of Fitparam() objects.
    X : array-like
        1-D or 2-D array containing the voltage (1st column) and if specified the Gfrac (2nd column) values.
    y : array-like
        1-D array containing the current values.
    T : float, optional
        Temperature in K, by default 300.
    exp_format : str or list of str, optional
        Format of the experimental data, by default 'light'.
    metric : str or list of str, optional
        Metric to evaluate the model, see optimpv.general.calc_metric for options, by default 'mse'.
    loss : str or list of str, optional
        Loss function to use, see optimpv.general.loss_function for options, by default 'linear'.
    threshold : int or list of int, optional
        Threshold value for the loss function used when doing multi-objective optimization, by default 100.
    minimize : bool or list of bool, optional
        If True then minimize the loss function, if False then maximize the loss function (note that if running a fit minize should be True), by default True.
    yerr : array-like or list of array-like, optional
        Errors in the current values, by default None.
    weight : array-like or list of array-like, optional
        Weights used for fitting if weight is None and yerr is not None, then weight = 1/yerr**2, by default None.
    tracking_metric : str or list of str, optional
        Additional metrics to track and report in run_Ax output, by default None.
    tracking_loss : str or list of str, optional
        Loss functions to apply to tracking metrics, by default None.
    tracking_exp_format : str or list of str, optional
        Experimental formats for tracking metrics, by default None.
    tracking_X : array-like or list of array-like, optional
        X values for tracking metrics, by default None.
    tracking_y : array-like or list of array-like, optional
        y values for tracking metrics, by default None.
    tracking_weight : array-like or list of array-like, optional
        Weights for tracking metrics, by default None.
    name : str, optional
        Name of the agent, by default 'diode'.
    use_pvlib : bool, optional
        If True then use the pvlib library to calculate the diode equation, by default False.
    **kwargs : dict
        Additional keyword arguments.
    """    
    def __init__(self, params, X, y, T = 300, exp_format = 'light', metric = 'mse', loss = 'linear', 
                threshold = 100, minimize = True, yerr = None, weight = None, 
                tracking_metric = None, tracking_loss = None, tracking_exp_format = None,
                tracking_X = None, tracking_y = None, tracking_weight = None,
                name = 'diode', use_pvlib = False, **kwargs):
        # super().__init__(**kwargs)

        self.params = params
        self.X = X # voltage and Gfrac
        self.y = y
        self.T = T # temperature in K
        
        # Convert single values to lists for consistency
        if isinstance(exp_format, str):
            self.exp_format = [exp_format]
        else:
            self.exp_format = exp_format
            
        if isinstance(metric, str):
            self.metric = [metric]
        else:
            self.metric = metric
            
        if isinstance(loss, str):
            self.loss = [loss]
        else:
            self.loss = loss
            
        if isinstance(threshold, (int, float)):
            self.threshold = [threshold]
        else:
            self.threshold = threshold
            
        if isinstance(minimize, bool):
            self.minimize = [minimize]
        else:
            self.minimize = minimize
            
        self.yerr = [yerr]
        
        # Handle weight calculation from yerr if needed
        if weight is None and yerr is not None:
            self.weight = [1/np.asarray(yerr)**2]
        else:
            self.weight = [weight]
            
        self.name = name
        self.use_pvlib = use_pvlib
        self.kwargs = kwargs
        
        # Initialize tracking parameters
        self.tracking_metric = tracking_metric
        self.tracking_loss = tracking_loss
        self.tracking_exp_format = tracking_exp_format
        self.tracking_X = tracking_X
        self.tracking_y = tracking_y
        self.tracking_weight = tracking_weight

        # Process tracking metrics and losses
        if self.tracking_metric is not None:
            if isinstance(self.tracking_metric, str):
                self.tracking_metric = [self.tracking_metric]
            
            if self.tracking_loss is None:
                self.tracking_loss = ['linear'] * len(self.tracking_metric)
            elif isinstance(self.tracking_loss, str):
                self.tracking_loss = [self.tracking_loss] * len(self.tracking_metric)
                
            # Ensure tracking_metric and tracking_loss have the same length
            if len(self.tracking_metric) != len(self.tracking_loss):
                raise ValueError('tracking_metric and tracking_loss must have the same length')
                
            # Process tracking_exp_format
            if self.tracking_exp_format is None:
                # Default to the main experiment formats if not specified
                self.tracking_exp_format = self.exp_format
            elif isinstance(self.tracking_exp_format, str):
                self.tracking_exp_format = [self.tracking_exp_format]
                
            # check that all elements in tracking_exp_format are valid
            for form in self.tracking_exp_format:
                if form not in ['dark', 'light']:
                    raise ValueError(f'{form} is an invalid tracking_exp_format, must be either "dark" or "light"')
            
            # Process tracking_X and tracking_y
            # Check if all tracking formats are in main exp_format
            all_formats_in_main = all(fmt in self.exp_format for fmt in self.tracking_exp_format)
            if self.tracking_X is None or self.tracking_y is None:
                
                if not all_formats_in_main:
                    raise ValueError('tracking_X and tracking_y must be provided when tracking_exp_format contains formats not in exp_format')
                
                # Construct tracking_X and tracking_y from main X and y based on matching formats
                self.tracking_X = []
                self.tracking_y = []
                
                for fmt in self.tracking_exp_format:
                    fmt_indices = [i for i, main_fmt in enumerate(self.exp_format) if main_fmt == fmt]
                    if fmt_indices:
                        # Use the first matching format's data
                        idx = fmt_indices[0]
                        self.tracking_X.append(self.X)
                        self.tracking_y.append(self.y)
            
            # Ensure tracking_X and tracking_y are lists
            if not isinstance(self.tracking_X, list):
                self.tracking_X = [self.tracking_X]
            if not isinstance(self.tracking_y, list):
                self.tracking_y = [self.tracking_y]
                
            # Check that tracking_X and tracking_y have the right lengths
            if len(self.tracking_X) != len(self.tracking_exp_format) or len(self.tracking_y) != len(self.tracking_exp_format):
                raise ValueError('tracking_X and tracking_y must have the same length as tracking_exp_format')
            
            # Process tracking_weight
            if self.tracking_weight is None and all_formats_in_main:
                # Use the main weights if available
                self.tracking_weight = []
                if all_formats_in_main:
                    for fmt in self.tracking_exp_format:
                        fmt_indices = [i for i, main_fmt in enumerate(self.exp_format) if main_fmt == fmt]
                        if fmt_indices:
                            idx = fmt_indices[0]
                            self.tracking_weight.append(self.weight[idx])
                        else:
                            self.tracking_weight.append(None)
                else:
                    self.tracking_weight = [None] * len(self.tracking_exp_format)
            elif not isinstance(self.tracking_weight, list):
                self.tracking_weight = [self.tracking_weight]
                
            # Ensure tracking_weight has the right length
            if len(self.tracking_weight) != len(self.tracking_exp_format):
                raise ValueError('tracking_weight must have the same length as tracking_exp_format')

        # check that all elements in exp_format are valid
        for form in self.exp_format:
            if form not in ['dark','light']:
                raise ValueError(f'{form} is an invalid exp_format, must be either "dark" or "light"')
                
        # Add compare_type parameter
        self.compare_type = self.kwargs.get('compare_type', 'linear')
        if 'compare_type' in self.kwargs.keys():
            self.kwargs.pop('compare_type')
            
        # Validate compare_type
        if self.compare_type not in ['linear', 'log', 'normalized', 'normalized_log', 'sqrt']:
            raise ValueError('compare_type must be either linear, log, normalized, normalized_log, or sqrt')

        if got_pvlib == False:
            self.use_pvlib = False
    
    def run(self,parameters):
        """Run the diode model and calculate the loss function

        Parameters
        ----------
        parameters : dict
            Dictionary of parameter names and values.

        Returns
        -------
        float
            Loss function value.
        """    

        # check that all the arguments are in the parameters dictionary
        arg_names = ['J0','n','R_series','R_shunt']
        if self.exp_format[0] == 'light':
            arg_names.append('Jph')

        # for arg in arg_names:
        #     if arg not in parameters.keys():
        #         raise ValueError('Parameter: {} not in parameters dictionary'.format(arg))
        
        if 'T' not in parameters.keys():
            T_ = self.T
        else:
            T_ = parameters['T']

        parameters_rescaled = self.params_rescale(parameters, self.params)
        
        if self.use_pvlib and got_pvlib:
            print('Using pvlib to calculate diode equation')
            nVt = parameters_rescaled['n']*kb*T_
            if self.exp_format[0] == 'dark':
                J = -i_from_v(self.X, 0, parameters_rescaled['J0'], parameters_rescaled['R_series'], parameters_rescaled['R_shunt'], nVt)
            elif self.exp_format[0] == 'light':
                J = -i_from_v(self.X, parameters_rescaled['Jph'], parameters_rescaled['J0'], parameters_rescaled['R_series'], parameters_rescaled['R_shunt'], nVt)
        else:
            if self.exp_format[0] == 'dark':
                J = NonIdealDiode_dark(self.X, parameters_rescaled['J0'], parameters_rescaled['n'], parameters_rescaled['R_series'], parameters_rescaled['R_shunt'], T = T_)

            elif self.exp_format[0] == 'light':
                J = NonIdealDiode_light(self.X, parameters_rescaled['J0'], parameters_rescaled['n'], parameters_rescaled['R_series'], parameters_rescaled['R_shunt'], parameters_rescaled['Jph'], T = T_)

        return J
    

    def run_Ax(self,parameters):
        """Run the diode model and calculate the loss function

        Parameters
        ----------
        parameters : dict
            Dictionary of parameter names and values.

        Returns
        -------
        float
            Loss function value.
        """    
        
        if self.exp_format[0] == 'light':   
            self.compare_logs = self.kwargs.get('compare_logs',False)
        else:
            self.compare_logs = self.kwargs.get('compare_logs',True)
        
        yfit = self.run(parameters) # run the diode model

        dum_dict = {}
        
        for i in range(len(self.exp_format)):
            metric_name = self.metric[i]
            
            # Apply data transformation based on compare_type
            if self.compare_type == 'linear' and self.compare_logs:
                epsilon = np.finfo(np.float64).eps
                # if 0 in yfit, then add epsilon to avoid log(0)
                yfit_trans = yfit.copy()
                yfit_trans[abs(yfit_trans) <= epsilon] = epsilon
                y_trans = copy.deepcopy(self.y)
                y_trans[abs(y_trans) <= epsilon] = epsilon
                
                metric_value = calc_metric(np.log10(abs(y_trans)), np.log10(abs(yfit_trans)), 
                                          sample_weight=self.weight[i], metric_name=metric_name)
            elif self.compare_type != 'linear':
                # Use the transform_data function for non-linear transforms
                y_trans, yfit_trans = transform_data(
                    self.y, yfit, transform_type=self.compare_type
                )
                metric_value = calc_metric(y_trans, yfit_trans, 
                                          sample_weight=self.weight[i], metric_name=metric_name)
            else:
                # Standard linear comparison
                metric_value = calc_metric(self.y, yfit, 
                                          sample_weight=self.weight[i], metric_name=metric_name)
            
            dum_dict[self.name+'_'+self.exp_format[i]+'_'+self.metric[i]] = loss_function(metric_value, loss=self.loss[i])
            
            # Calculate tracking metrics if they exist
            if self.tracking_metric is not None:
                for j in range(len(self.tracking_metric)):
                    if self.tracking_exp_format[j] == self.exp_format[i]:
                        tracking_metric_name = self.tracking_metric[j]
                        
                        # Apply the same transform as the main metric
                        if self.compare_type == 'linear' and self.compare_logs:
                            tracking_metric_value = calc_metric(np.log10(abs(y_trans)), np.log10(abs(yfit_trans)), 
                                                             sample_weight=self.tracking_weight[j], metric_name=tracking_metric_name)
                        elif self.compare_type != 'linear':
                            tracking_metric_value = calc_metric(y_trans, yfit_trans, 
                                                             sample_weight=self.tracking_weight[j], metric_name=tracking_metric_name)
                        else:
                            tracking_metric_value = calc_metric(self.tracking_y[j], yfit, 
                                                             sample_weight=self.tracking_weight[j], metric_name=tracking_metric_name)
                        
                        dum_dict[self.name+'_'+self.tracking_exp_format[j]+'_tracking_'+tracking_metric_name] = loss_function(
                            tracking_metric_value, loss=self.tracking_loss[j])

        return dum_dict