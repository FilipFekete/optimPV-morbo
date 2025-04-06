"""axBOtorchOptimizer module. This module contains the axBOtorchOptimizer class. The class is used to run the bayesian optimization process using the Ax library."""
######### Package Imports #########################################################################
from dataclasses import dataclass
import torch
from botorch.acquisition import qExpectedImprovement, qLogExpectedImprovement
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.fit import fit_gpytorch_mll
from botorch.generation import MaxPosteriorSampling
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.test_functions import Ackley
from botorch.utils.transforms import unnormalize
from torch.quasirandom import SobolEngine

import gpytorch
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition.objective import ScalarizedPosteriorTransform

import math, copy
import numpy as np
from joblib import Parallel, delayed
from functools import partial
from optimpv import *
from optimpv.axBOtorch.axUtils import *
from optimpv.axBOtorch.axUtils import *
from optimpv.axBOtorch.axSchedulerUtils import *
import ax, os, shutil
from ax import *
from ax.service.ax_client import AxClient
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax import Models
from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.service.scheduler import Scheduler, SchedulerOptions, TrialType
from collections import defaultdict
from torch.multiprocessing import Pool, set_start_method
# from multiprocessing import Pool, set_start_method
# try: # needed for multiprocessing when using pytorch
set_start_method('spawn',force=True)
# except RuntimeError:
#     print("spawn method already set")
#     pass

from logging import Logger
from ax.utils.common.logger import get_logger, _round_floats_for_logging

logger: Logger = get_logger(__name__)
ROUND_FLOATS_IN_LOGS_TO_DECIMAL_PLACES: int = 6
round_floats_for_logging = partial(
    _round_floats_for_logging,
    decimal_places=ROUND_FLOATS_IN_LOGS_TO_DECIMAL_PLACES,
)

from optimpv.general.BaseAgent import BaseAgent
from optimpv.posterior.posterior import get_df_from_ax

######### Optimizer Definition #######################################################################
class axBOtorchOptimizer(BaseAgent):
    """Initialize the axBOtorchOptimizer class. The class is used to run the optimization process using the Ax library. 

    Parameters
    ----------
    params : list of Fitparam() objects, optional
        List of Fitparam() objects, by default None
    agents : list of Agent() objects, optional
        List of Agent() objects see optimpv/general/BaseAgent.py for a base class definition, by default None
    models : list, optional
        list of models to use for the optimization process, by default ['SOBOL','BOTORCH_MODULAR']
    n_batches : list, optional
        list of the number of batches for each model, by default [1,10]
    batch_size : list, optional
        list of the batch sizes for each model, by default [10,2]
    ax_client : AxClient, optional
        AxClient object, by default None
    max_parallelism : int, optional
        maximum number of parallel processes to run, by default 10
    model_kwargs_list : dict, optional
        dictionary of model kwargs for each model, by default None
    model_gen_kwargs_list : dict, optional
        dictionary of model generation kwargs for each model, by default None
    name : str, optional
        name of the optimization process, by default 'ax_opti'

    Raises
    ------
    ValueError
        raised if the number of batches and the number of models are not the same
    ValueError
        raised if the model is not a string or a Models enum
    ValueError
        raised if the model_kwargs_list and models do not have the same length
    ValueError
        raised if the model_gen_kwargs_list and models do not have the same length
    """ 
    def __init__(self, params = None, agents = None, models = ['SOBOL','BOTORCH_MODULAR'],n_batches = [1,10], batch_size = [10,2], ax_client = None,  max_parallelism = -1,model_kwargs_list = None, model_gen_kwargs_list = None, name = 'ax_opti', **kwargs):
               
        self.params = params
        if not isinstance(agents, list):
            agents = [agents]
        self.agents = agents
        self.models = models
        self.n_batches = n_batches
        self.batch_size = batch_size
        self.all_metrics = None
        self.all_tracking_metrics = None
        self.ax_client = ax_client
        self.max_parallelism = max_parallelism
        if max_parallelism == -1:
            self.max_parallelism = os.cpu_count()-1
        if model_kwargs_list is None:
            model_kwargs_list = [{}]*len(models)
        elif isinstance(model_kwargs_list,dict):
            model_kwargs_list = [model_kwargs_list]*len(models)
        elif len(model_kwargs_list) != len(models):
            raise ValueError('model_kwargs_list must have the same length as models')
        self.model_kwargs_list = model_kwargs_list
        if model_gen_kwargs_list is None:
            model_gen_kwargs_list = [{}]*len(models)
        elif isinstance(model_gen_kwargs_list,dict):
            model_gen_kwargs_list = [model_gen_kwargs_list]*len(models) 
        elif len(model_gen_kwargs_list) != len(models):
            raise ValueError('model_gen_kwargs_list must have the same length as models')
        self.model_gen_kwargs_list = model_gen_kwargs_list
        self.name = name
        self.kwargs = kwargs
        
        if len(n_batches) != len(models):
            raise ValueError('n_batches and models must have the same length')
        if type(batch_size) == int:
            self.batch_size = [batch_size]*len(models)
        if len(batch_size) != len(models):
            raise ValueError('batch_size and models must have the same length')


    def create_generation_strategy(self):
        """ Create a generation strategy for the optimization process using the models and the number of batches and batch sizes. See ax documentation for more details: https://ax.dev/tutorials/generation_strategy.html

        Returns
        -------
        GenerationStrategy
            The generation strategy for the optimization process

        Raises
        ------
        ValueError
            If the model is not a string or a Models enum
        """        

        steps = []
        for i, model in enumerate(self.models):
            if type(model) == str:
                model = Models[model]
            elif isinstance(model, Models):
                model = model
            else:
                raise ValueError('Model must be a string or a Models enum')

            steps.append(GenerationStep(
                model=model,
                num_trials=self.n_batches[i]*self.batch_size[i],
                max_parallelism=min(self.max_parallelism,self.batch_size[i]),
                model_kwargs= self.model_kwargs_list[i],
                model_gen_kwargs= self.model_gen_kwargs_list[i],
            ))

        gs = GenerationStrategy(steps=steps, )

        return gs

    def create_generation_strategy_batch(self):
        """ Create a generation strategy for the optimization process using the models and the number of batches and batch sizes. See ax documentation for more details: https://ax.dev/tutorials/generation_strategy.html

        Returns
        -------
        GenerationStrategy
            The generation strategy for the optimization process

        Raises
        ------
        ValueError
            If the model is not a string or a Models enum
        """        

        steps = []
        for i, model in enumerate(self.models):
            if type(model) == str:
                model = Models[model]
            elif isinstance(model, Models):
                model = model
            else:
                raise ValueError('Model must be a string or a Models enum')
            # check if "n" in model_gen_kwargs_list[i]:
            if 'n' in self.model_gen_kwargs_list[i]:
                self.model_gen_kwargs_list[i]['n'] = self.batch_size[i]
            if 'n' in self.model_kwargs_list[i]:
                self.model_kwargs_list[i]['n'] = self.batch_size[i]
            steps.append(GenerationStep(
                model=model,
                num_trials=self.n_batches[i],#*self.batch_size[i],
                max_parallelism=min(self.max_parallelism,self.batch_size[i]),
                model_kwargs= self.model_kwargs_list[i],
                model_gen_kwargs= self.model_gen_kwargs_list[i],
            ))

        gs = GenerationStrategy(steps=steps, )

        return gs
    
    def get_tracking_metrics(self, agents):
        """ Extract tracking metrics from agents
        
        Parameters
        ----------
        agents : list
            List of Agent objects
            
        Returns
        -------
        list
            List of tracking metric names formatted with agent name prefix
        """
        tracking_metrics = []
        for agent in agents:
            if hasattr(agent, 'tracking_metric') and agent.tracking_metric is not None:
                # Check if agent has tracking_exp_format attribute (like RateEqAgent)
                if hasattr(agent, 'tracking_exp_format') and agent.tracking_exp_format is not None:
                    # Use experiment format in the metric name
                    for i in range(len(agent.tracking_metric)):
                        exp_fmt = agent.tracking_exp_format[i] if i < len(agent.tracking_exp_format) else "unknown"
                        tracking_metric_name = agent.name+'_'+exp_fmt+'_tracking_'+agent.tracking_metric[i]
                        if tracking_metric_name not in tracking_metrics:
                            tracking_metrics.append(tracking_metric_name)
                else:
                    raise ValueError(f"Agent {agent.name} does not have tracking_exp_format attribute.")
        return tracking_metrics
    
    def create_objectives(self):
        """ Create the objectives for the optimization process. The objectives are the metrics of the agents. The objectives are created using the metric, minimize and threshold attributes of the agents. If the agent has an exp_format attribute, it is used to create the objectives.

        Returns
        -------
        dict
            A dictionary of the objectives for the optimization process
        """        

        append_metrics = False
        if self.all_metrics is None:
            self.all_metrics = []
            append_metrics = True
            
        objectives = {}
        for agent in self.agents:
            for i in range(len(agent.metric)):
                # if exp_format is an attribute of the agent, use it
                if hasattr(agent,'exp_format'):
                    objectives[agent.name+'_'+agent.exp_format[i]+'_'+agent.metric[i]] = ObjectiveProperties(minimize=agent.minimize[i], threshold=agent.threshold[i])
                    if append_metrics:
                        self.all_metrics.append(agent.name+'_'+agent.exp_format[i]+'_'+agent.metric[i])
                else:
                    objectives[agent.name+'_'+agent.metric[i]] = ObjectiveProperties(minimize=agent.minimize[i], threshold=agent.threshold[i])
                    if append_metrics:
                        self.all_metrics.append(agent.name+'_'+agent.metric[i])

        return objectives
    
    def evaluate(self,args):
        """ Evaluate the agent on a parameter point

        Parameters
        ----------
        args : tuple
            Tuple containing the index of the agent, the agent, the index of the parameter point and the parameter point

        Returns
        -------
        tuple
            Tuple containing the index of the parameter point and the results of the agent on the parameter point
        """        
        idx, agent, p_idx, p = args
        res = agent.run_Ax(p)
        return p_idx, res
    
    def optimize(self,batch=False):
        """ Run the optimization process using the agents and the parameters. The optimization process uses the Ax library. The optimization process runs the agents in parallel if the parallel_agents attribute is True. The optimization process runs using the parameters, agents, models, n_batches, batch_size, max_parallelism, model_kwargs_list, model_gen_kwargs_list, name and kwargs attributes of the class. The optimization process runs using the create_generation_strategy and create_objectives methods of the class. The optimization process runs using the run_Ax method of the agents.

        Parameters
        ----------
        batch : bool, optional
            If True, run the optimization process in batch mode. The default is False.

        Raises
        ------
        ValueError
            If the number of batches and the number of models are not the same

        """  
        if batch:
            self.optimize_batch()
        else:
            self.optimize_sequential()

    def optimize_sequential(self):
        """ Run the optimization process using the agents and the parameters. The optimization process uses the Ax library. The optimization process runs the agents in parallel if the parallel_agents attribute is True. The optimization process runs using the parameters, agents, models, n_batches, batch_size, max_parallelism, model_kwargs_list, model_gen_kwargs_list, name and kwargs attributes of the class. The optimization process runs using the create_generation_strategy and create_objectives methods of the class. The optimization process runs using the run_Ax method of the agents.

        Raises
        ------
        ValueError
            If the number of batches and the number of models are not the same

        """        

        # from kwargs
        enforce_sequential_optimization = self.kwargs.get('enforce_sequential_optimization',False)
        global_max_parallelism = self.kwargs.get('global_max_parallelism',-1)
        verbose_logging = self.kwargs.get('verbose_logging',True)
        global_stopping_strategy = self.kwargs.get('global_stopping_strategy',None)
        outcome_constraints = self.kwargs.get('outcome_constraints',None)
        parameter_constraints = self.kwargs.get('parameter_constraints',None)
        parallel_agents = self.kwargs.get('parallel_agents',True)

        # create parameters space from params
        parameters_space = ConvertParamsAx(self.params)

        # Get tracking metrics directly
        self.all_tracking_metrics = self.get_tracking_metrics(self.agents)

        # create generation strategy
        gs = self.create_generation_strategy()

        # create ax client
        if self.ax_client is None:
            self.ax_client = AxClient(generation_strategy=gs, enforce_sequential_optimization=enforce_sequential_optimization, verbose_logging=verbose_logging,global_stopping_strategy=global_stopping_strategy)
        
        # create experiment
        self.ax_client.create_experiment(
            name=self.name,
            parameters=parameters_space,
            objectives=self.create_objectives(),
            outcome_constraints=outcome_constraints,
            parameter_constraints=parameter_constraints,
            tracking_metric_names=self.all_tracking_metrics,
        )

        # run optimization
        num = 0
        total_trials = sum(np.asarray(self.n_batches)*np.asarray(self.batch_size))
        n_step_points = np.cumsum(np.asarray(self.n_batches)*np.asarray(self.batch_size))
        size_pool = None
        while num < total_trials:
            # check the current batch size
            curr_batch_size = self.batch_size[np.argmax(n_step_points>num)]
            num += curr_batch_size
            if num > total_trials:
                curr_batch_size = curr_batch_size - (num-total_trials)

            parameters, trial_index = self.ax_client.get_next_trials(curr_batch_size)
            
            if not parallel_agents:
                results = []
                for idx, agent in enumerate(self.agents):
                    dum_res = Parallel(n_jobs=min(curr_batch_size*len(self.agents),self.max_parallelism))(delayed(agent.run_Ax)(p) for p in parameters.values())
                    results.append(dum_res)
                
                main_results = []
                # merge the n agents results
                for i in range(len(results[0])):
                    main_results.append({})
                    for j in range(len(results)):
                        main_results[-1].update(results[j][i])
            else:
                agent_param_list =[]
                for p_idx, p in enumerate(parameters.values()):
                    for idx, agent in enumerate(self.agents):
                        agent_param_list.append((idx, agent, p_idx, p))

                # Run all combinations in parallel using multiprocessing
                if size_pool is None:
                    size_pool = min(len(agent_param_list),self.max_parallelism)
                if size_pool != min(len(agent_param_list),self.max_parallelism):
                    # close the old pool
                    size_pool = min(len(agent_param_list),self.max_parallelism)
                    pool.close()
                    pool.join()
                    
                # Run all combinations in parallel using multiprocessing
                with Pool(processes=min(len(agent_param_list),self.max_parallelism)) as pool:
                    parallel_results = pool.map(self.evaluate, agent_param_list)

                # Collect and merge results
                results_dict = defaultdict(dict)
                for p_idx, res in parallel_results:
                    results_dict[p_idx].update(res)

                # Convert to main_results list
                main_results = [results_dict[i] for i in sorted(results_dict)]

            for trial_index, raw_data in zip(parameters.keys(), main_results):
                got_nan = False
                for key in raw_data.keys():
                    if np.isnan(raw_data[key]):
                        got_nan = True
                        break
                if not got_nan:
                    self.ax_client.complete_trial(trial_index, raw_data=raw_data)
                else:
                    self.ax_client.log_trial_failure(trial_index)


    def optimize_batch(self):
        """ Run the optimization process using the agents and the parameters. The optimization process uses the Ax library. The optimization process runs the agents in parallel if the parallel_agents attribute is True. The optimization process runs using the parameters, agents, models, n_batches, batch_size, max_parallelism, model_kwargs_list, model_gen_kwargs_list, name and kwargs attributes of the class. The optimization process runs using the create_generation_strategy and create_objectives methods of the class. The optimization process runs using the run_Ax method of the agents.

        Raises
        ------
        ValueError
            If the number of batches and the number of models are not the same

        """

        # from kwargs
        enforce_sequential_optimization = self.kwargs.get('enforce_sequential_optimization',False)
        global_max_parallelism = self.kwargs.get('global_max_parallelism',-1)
        verbose_logging = self.kwargs.get('verbose_logging',True)
        scheduler_logging_level = self.kwargs.get('scheduler_logging_level',0)
        global_stopping_strategy = self.kwargs.get('global_stopping_strategy',None)
        outcome_constraints = self.kwargs.get('outcome_constraints',None)
        parameter_constraints = self.kwargs.get('parameter_constraints',None)
        parallel_agents = self.kwargs.get('parallel_agents',True)
        max_number_cores = self.kwargs.get('max_number_cores',-1)
        init_seconds_between_polls = self.kwargs.get('init_seconds_between_polls',0.1)
        logging_level = self.kwargs.get('logging_level',20)
        keep_tmp_dir = self.kwargs.get('keep_tmp_dir',False)

        if max_number_cores == -1:
            max_number_cores = os.cpu_count()-1
        tmp_dir = self.kwargs.get('tmp_dir',None)
        tmp_dir = os.path.join(os.getcwd(),'.tmp_dir') if tmp_dir is None else tmp_dir

        # create parameters space from params
        parameters_space = ConvertParamsAx(self.params)

        # Get tracking metrics directly
        self.all_tracking_metrics = self.get_tracking_metrics(self.agents)

        # create generation strategy
        gs = self.create_generation_strategy_batch()

        # create ax client
        if self.ax_client is None:
            self.ax_client = AxClient(generation_strategy=gs, enforce_sequential_optimization=enforce_sequential_optimization, verbose_logging=verbose_logging,global_stopping_strategy=global_stopping_strategy)
        
        _obj = self.create_objectives()

        is_multi_obj = False
        if len(_obj.keys()) > 1:
            is_multi_obj = True

        q = Pool(max_number_cores)
        if not is_multi_obj:
            # obj = Objective(metric=MockJobMetric(name=list(_obj.keys())[0]+'_', agents = self.agents, pool = q, tmp_dir = tmp_dir, parallel_agents = parallel_agents), minimize=True)
            obj = Objective(metric=MockJobMetric(name=list(_obj.keys())[0], agents = self.agents, pool = q, tmp_dir = tmp_dir, parallel_agents = parallel_agents), minimize=_obj[list(_obj.keys())[0]].minimize)
        else:
            objectives_list = []
            objectives_thresholds = []

            for key in _obj.keys():
                lower_is_better = _obj[key].minimize
                metric = MockJobMetric(name=key, agents = self.agents, pool = q, tmp_dir = tmp_dir, parallel_agents = parallel_agents,lower_is_better=lower_is_better)
                objectives_list.append(Objective(metric=metric, minimize=lower_is_better))
                objectives_thresholds.append(ObjectiveThreshold(metric=metric, bound=_obj[key].threshold,relative=False))
            obj = MultiObjective(objectives=objectives_list, objective_thresholds=objectives_thresholds)
            # raise ValueError('The objective must be a single metric')

        # create experiment
        self.ax_client.create_experiment(
            name=self.name,
            parameters=parameters_space,
            # objectives=self.create_objectives(),
            outcome_constraints=outcome_constraints,
            parameter_constraints=parameter_constraints,
            tracking_metric_names=self.all_tracking_metrics,
        )
        # threshold=
        if not is_multi_obj:
            self.ax_client.experiment.optimization_config=OptimizationConfig(objective=obj)
        else:
            self.ax_client.experiment.optimization_config=MultiObjectiveOptimizationConfig(objective=obj)
            # self.ax_client.experiment.optimization_config.objective_thresholds = objectives_thresholds
        
        # create runner
        runner = MockJobRunner(agents = self.agents, pool = q, tmp_dir = tmp_dir, parallel_agents = parallel_agents)
        self.ax_client.experiment.runner = runner
        # run optimization
        num = 0
        total_trials = sum(np.asarray(self.n_batches)*np.asarray(self.batch_size))
        n_step_points = np.cumsum(np.asarray(self.n_batches)*np.asarray(self.batch_size))

        if verbose_logging:
            logger.info('Starting optimization with %d batches and a total of %d trials',sum(np.asarray(self.n_batches)),total_trials)

        count = 1
        while num < total_trials:
            if verbose_logging and num != 0:
                logging_level = 20
                logger.setLevel(logging_level)
                logger.info(f'Starting batch {round_floats_for_logging(count)} with {round_floats_for_logging(self.batch_size[np.argmax(n_step_points>num)])} trials')
            # check the current batch size
            if num == 0:
                old_batch_size = self.batch_size[np.argmax(n_step_points>num)]
            else:
                old_batch_size = curr_batch_size

            curr_batch_size = self.batch_size[np.argmax(n_step_points>num)]

            if old_batch_size != curr_batch_size or num == 0: # if the batch size changes, create a new scheduler
                # Create a new scheduler for each batch with the current batch size
                scheduler = Scheduler(
                    experiment=self.ax_client.experiment,
                    generation_strategy=self.ax_client.generation_strategy,
                    options=SchedulerOptions(run_trials_in_batches=True,init_seconds_between_polls=init_seconds_between_polls,trial_type=TrialType.BATCH_TRIAL,batch_size=curr_batch_size,logging_level=scheduler_logging_level,global_stopping_strategy=global_stopping_strategy),
                )

            num += curr_batch_size
            if num > total_trials:
                curr_batch_size = curr_batch_size - (num-total_trials)
            
            scheduler.run_n_trials(max_trials=1)
            if verbose_logging:
                logging_level = 20
                logger.setLevel(logging_level)
                logger.info('Finished batch %d', count)
            count += 1
              
        q.close()
        q.join()
        if verbose_logging:
            logging_level = 20
            logger.setLevel(logging_level)
            logger.info('Finished optimization')

        # clean up the tmp_dir
        if not keep_tmp_dir:
            if os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir)

    def optimize_turbo(self,acq_turbo='ts',force_continue = False, kwargs_turbo_state={},kwargs_turbo={}):
        """Run the optimzation using Turbo. This is based on the Botorch implementation of Turbo. See https://botorch.org/docs/tutorials/turbo_1/ for more details.

        Parameters
        ----------
        acq_turbo : str, optional
            The acquisition function to use can be 'ts' or 'ei', by default 'ts'
        force_continue : bool, optional
            If True, the optimization will continue even if a restart is triggered, by default False
        kwargs_turbo_state : dict, optional
            The kwargs to use for the TurboState, by default {}
            can be: 
            - length: float, by default 0.8
            - length_min: float, by default 0.5**7
            - length_max: float, by default 1.6
            - success_tolerance: int, by default 10
        kwargs_turbo : dict, optional
            The kwargs to use for the Turbo, by default {}

        Raises
        ------
        ValueError
            Turbo does not support parameter constraints
        ValueError
            Turbo does not support outcome constraints
        ValueError
            Turbo only supports single objective optimization
        ValueError
            Turbo only supports 2 models
        ValueError
            Turbo only supports Sobol as the first model
        ValueError
            Turbo only supports BoTorch as the second model
        """            

        parameters_space = ConvertParamsAx(self.params)
        objectives=self.create_objectives()

        # Get tracking metrics directly
        self.all_tracking_metrics = self.get_tracking_metrics(self.agents)

        # make sure that we do not take fixed params into account
        free_pnames = [p['name'] for p in parameters_space if p['type'] != 'fixed']
        dim = len(free_pnames)

        parallel_agents = self.kwargs.get('parallel_agents',True)
        verbose_logging = self.kwargs.get('verbose_logging',True)
        enforce_sequential_optimization = self.kwargs.get('enforce_sequential_optimization',False)
        global_stopping_strategy = self.kwargs.get('global_stopping_strategy',None)
        outcome_constraints = self.kwargs.get('outcome_constraints',None)
        parameter_constraints = self.kwargs.get('parameter_constraints',None)
        # acq_turbo = self.kwargs.get('acq_turbo','ts')
        # kwargs_turbo_state = self.kwargs.get('kwargs_turbo_state',{})
        NUM_RESTARTS = kwargs_turbo.get('NUM_RESTARTS', 10)
        RAW_SAMPLES = kwargs_turbo.get('RAW_SAMPLES', 512)
        N_CANDIDATES = kwargs_turbo.get('N_CANDIDATES', min(5000, max(2000, 200 * dim)))

        

        if parameter_constraints is not None:
            raise ValueError('Turbo does not support parameter constraints')
        if outcome_constraints is not None:
            raise ValueError('Turbo does not support outcome constraints')  
        
        # check if we have a single objective
        if len(objectives) > 1:
            raise ValueError('Turbo only supports single objective optimization')
        # check if we minimize
        minimize = list(objectives.values())[0].minimize
        if minimize:
            fac = -1
        else:
            fac = 1

        if len(self.models)>2:
            raise ValueError('Turbo only supports 2 models')
        if self.models[0] != 'SOBOL':
            raise ValueError('Turbo only supports Sobol as the first model')
        if self.models[1] != 'BOTORCH_MODULAR':
            raise ValueError('Turbo only supports BoTorch as the second model')
        
        # Set the device and dtype
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.double
        max_cholesky_size = float("inf")  # Always use Cholesky
        
        total_trials = sum(np.asarray(self.n_batches)*np.asarray(self.batch_size))
        if verbose_logging:
            logger.info('Starting optimization with %d batches and a total of %d trials',sum(np.asarray(self.n_batches)),total_trials)

        # Start with a Sobol sequence
        n_total_sobol = self.n_batches[0]*self.batch_size[0]
        num_sobol = 0
        bounds = torch.tensor([p['bounds'] for p in parameters_space if p['type'] != 'fixed'], device=device, dtype=dtype)

        # transpose bounds
        bounds = bounds.transpose(0,1)
        # Create and run initial points per batch
        count_batch = 1
        
        while num_sobol < n_total_sobol:
            if verbose_logging:
                logging_level = 20
                logger.setLevel(logging_level)
                logger.info('Starting Sobol batch %d with %d trials', count_batch, self.batch_size[0])
            
            # Get initial points
            X_turbo = get_initial_points(
                dim=dim,
                n_pts=self.batch_size[0],
                device=device,
                dtype=dtype,
            )
            
            # unnormalize
            X_turbo_un = unnormalize(X_turbo, bounds=bounds)
            # build list of dicts with noam
            dics = []
            for p in X_turbo_un:
                p = p.cpu().numpy()
                # write p into a dict with the param names as keys if the param is not fixed
                p = {self.params[i].name: p[i] for i in range(len(self.params)) if self.params[i].type != 'fixed'}
                dics.append(p)

            # run agents 
            if parallel_agents:
                agent_param_list =[]
                for p_idx, p in enumerate(dics):
                    for idx, agent in enumerate(self.agents):
                        agent_param_list.append((idx, agent, p_idx, p))

                # Run all combinations in parallel using multiprocessing
                with Pool(processes=min(len(agent_param_list),self.max_parallelism)) as pool:
                    parallel_results = pool.map(self.evaluate, agent_param_list)

                # Collect and merge results
                results_dict = defaultdict(dict)
                for p_idx, res in parallel_results:
                    results_dict[p_idx].update(res)

                # Convert to main_results list
                main_results = [results_dict[i] for i in sorted(results_dict)]
            else:
                results = []
                for idx, agent in enumerate(self.agents):
                    dum_res = Parallel(n_jobs=min(len(dics),self.max_parallelism))(delayed(agent.run_Ax)(p) for p in dics)
                    results.append(dum_res)
                
                main_results = []
                # merge the n agents results
                for i in range(len(results[0])):
                    main_results.append({})
                    for j in range(len(results)):
                        main_results[-1].update(results[j][i])
            
            # Only keep values from result dictionary that are in all_metrics
            Y_turbo = torch.tensor([[res[metric] for metric in self.all_metrics] for res in main_results], device=device, dtype=dtype)
            # multiplication factor
            Y_turbo = fac*Y_turbo
            
            # Also collect tracking metrics if they exist
            Y_tracking = None
            if self.all_tracking_metrics and len(self.all_tracking_metrics) > 0:
                tracking_data = []
                for res in main_results:
                    metrics_vals = []
                    for metric in self.all_tracking_metrics:
                        if metric in res:
                            metrics_vals.append(res[metric])
                        else:
                            metrics_vals.append(float('nan'))
                    tracking_data.append(metrics_vals)
                if tracking_data:
                    Y_tracking = torch.tensor(tracking_data, device=device, dtype=dtype)
                    
            num_sobol += self.batch_size[0]
            count_batch += 1

        if verbose_logging:
            logging_level = 20
            logger.setLevel(logging_level)
            logger.info('Finished Sobol')
        
        # Create a new state for each batch
        best_value = max(Y_turbo).item()
        state = TurboState(dim=dim, batch_size=self.batch_size[1], best_value=best_value,**kwargs_turbo_state)
        max_num_trials = self.n_batches[1]*self.batch_size[1]
        num_turbo = 0
        
        while (not num_turbo > max_num_trials) and not (state.restart_triggered and not force_continue):
            if verbose_logging:
                logging_level = 20
                logger.setLevel(logging_level)
                if state.restart_triggered and force_continue:
                    logger.setLevel(logging_level)
                    logger.info('Restart triggered, but we force the optimization to continue.')
            try:
                # Fit a GP model
                train_Y = (Y_turbo - Y_turbo.mean()) / Y_turbo.std()

                try:
                    likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
                    covar_module = ScaleKernel(  # Use the same lengthscale prior as in the TuRBO paper
                        MaternKernel(
                            nu=2.5, ard_num_dims=dim, lengthscale_constraint=Interval(0.005, 4.0)
                        )
                    )
                    model = SingleTaskGP(
                        X_turbo, train_Y, covar_module=covar_module, likelihood=likelihood
                    )
                    mll = ExactMarginalLogLikelihood(model.likelihood, model)
                
                    # Do the fitting and acquisition function optimization inside the Cholesky context
                    with gpytorch.settings.max_cholesky_size(max_cholesky_size):
                        # Fit the model
                        fit_gpytorch_mll(mll)

                        # Create a batch
                        X_next = generate_batch(
                            state=state,
                            model=model,
                            X=X_turbo,
                            Y=train_Y,
                            batch_size=state.batch_size,
                            n_candidates=N_CANDIDATES,
                            num_restarts=NUM_RESTARTS,
                            raw_samples=RAW_SAMPLES,
                            acqf=acq_turbo,
                            device=device,
                            dtype=dtype,
                            minimize=minimize,
                        )
                except Exception as e:
                    # Fall back to a more robust likelihood with stronger regularization
                    likelihood = GaussianLikelihood(noise_constraint=Interval(1e-4, 0.1))
                    covar_module = ScaleKernel(  # Use the same lengthscale prior as in the TuRBO paper
                        MaternKernel(
                            nu=2.5, ard_num_dims=dim, lengthscale_constraint=Interval(0.005, 4.0)
                        )
                    )
                    model = SingleTaskGP(X_turbo, train_Y, covar_module=covar_module, likelihood=likelihood)
                    mll = ExactMarginalLogLikelihood(model.likelihood, model)
                    with gpytorch.settings.max_cholesky_size(max_cholesky_size):
                        # Fit the model
                        fit_gpytorch_mll(mll)

                        # Create a batch
                        X_next = generate_batch(
                            state=state,
                            model=model,
                            X=X_turbo,
                            Y=train_Y,
                            batch_size=state.batch_size,
                            n_candidates=N_CANDIDATES,
                            num_restarts=NUM_RESTARTS,
                            raw_samples=RAW_SAMPLES,
                            acqf=acq_turbo,
                            device=device,
                            dtype=dtype,
                            minimize=minimize,
                        )
                    logging_level = 20
                    logger.setLevel(logging_level)
                    logger.error(f"Error in Turbo batch {count_batch}: {e}")
                    logger.error(f"We are stopping the optimization process")
                    break

                # Evaluate the batch
                X_next_un = unnormalize(X_next, bounds=bounds)
                # build list of dicts with noam
                dics = []
                for p in X_next_un:
                    p = p.cpu().numpy()
                    # write p into a dict with the param names as keys
                    p = {self.params[i].name: p[i] for i in range(len(self.params)) if self.params[i].type != 'fixed'}
                    dics.append(p)
                # run agents
                if parallel_agents:
                    agent_param_list =[]
                    for p_idx, p in enumerate(dics):
                        for idx, agent in enumerate(self.agents):
                            agent_param_list.append((idx, agent, p_idx, p))

                    # Run all combinations in parallel using multiprocessing
                    with Pool(processes=min(len(agent_param_list),self.max_parallelism)) as pool:
                        parallel_results = pool.map(self.evaluate, agent_param_list)

                    # Collect and merge results
                    results_dict = defaultdict(dict)
                    for p_idx, res in parallel_results:
                        results_dict[p_idx].update(res)

                    # Convert to main_results list
                    main_results = [results_dict[i] for i in sorted(results_dict)]
                else:
                    results = []
                    for idx, agent in enumerate(self.agents):
                        dum_res = Parallel(n_jobs=min(len(dics),self.max_parallelism))(delayed(agent.run_Ax)(p) for p in dics)
                        results.append(dum_res)
                    
                    main_results = []
                    # merge the n agents results
                    for i in range(len(results[0])):
                        main_results.append({})
                        for j in range(len(results)):
                            main_results[-1].update(results[j][i])

                # Only keep values from result dictionary that are in all_metrics
                Y_next = torch.tensor([[res[metric] for metric in self.all_metrics] for res in main_results], device=device, dtype=dtype)
                # multiplication factor
                Y_next = fac*Y_next
                
                # Also collect tracking metrics if they exist
                Y_next_tracking = None
                if self.all_tracking_metrics and len(self.all_tracking_metrics) > 0:
                    tracking_data = []
                    for res in main_results:
                        metrics_vals = []
                        for metric in self.all_tracking_metrics:
                            if metric in res:
                                metrics_vals.append(res[metric])
                            else:
                                metrics_vals.append(float('nan'))
                        tracking_data.append(metrics_vals)
                    if tracking_data:
                        Y_next_tracking = torch.tensor(tracking_data, device=device, dtype=dtype)
                
                # Update state
                state = update_state(state=state, Y_next=Y_next)

                # Append data
                X_turbo = torch.cat((X_turbo, X_next), dim=0)
                Y_turbo = torch.cat((Y_turbo, Y_next), dim=0)
                if Y_tracking is not None and Y_next_tracking is not None:
                    Y_tracking = torch.cat((Y_tracking, Y_next_tracking), dim=0)
                elif Y_next_tracking is not None:
                    Y_tracking = Y_next_tracking
                    
                num_turbo += state.batch_size
                
                # Print current status
                if verbose_logging:
                    logging_level = 20
                    logger.setLevel(logging_level)
                    logger.info(f"Finished Turbo batch {count_batch} with {state.batch_size} trials with current best value: {fac*state.best_value:.2e}, TR length: {state.length:.2e}")
                
                count_batch += 1
            except Exception as e:
                logging_level = 20
                logger.setLevel(logging_level)
                logger.error(f"Error in Turbo batch {count_batch}: {e}")
                logger.error(f"We are stopping the optimization process")
                break
                
            
        # load all data into ax
        

        # load all data into ax
        # create ax client
        # create generation strategy using the second model
        gs = [GenerationStep(
            model=Models[self.models[1]],
            num_trials=1,
            max_parallelism=min(self.max_parallelism,self.batch_size[1]),
            model_kwargs= self.model_kwargs_list[1],
            model_gen_kwargs= self.model_gen_kwargs_list[1],
        )]
        gs = GenerationStrategy(steps=gs, )

        # create ax client
        if self.ax_client is None:
            self.ax_client = AxClient(generation_strategy=gs, enforce_sequential_optimization=enforce_sequential_optimization, verbose_logging=verbose_logging,global_stopping_strategy=global_stopping_strategy)
        
        self.ax_client.create_experiment(
            name=self.name,
            parameters=parameters_space,
            objectives=self.create_objectives(),
            tracking_metric_names=self.all_tracking_metrics,
        )

        # add all data to ax
        X_turbo_un = unnormalize(X_turbo, bounds=bounds)
        for i in range(len(X_turbo_un)):
            dic = {}
            for j in range(len(X_turbo_un[i])):
                dic[free_pnames[j]] = X_turbo_un[i][j].item()
            # add fixed params to dic
            for p in self.params:
                if p.type == 'fixed':
                    dic[p.name] = p.value
            parameters, trial_index = self.ax_client.attach_trial(parameters=dic)
            # add all_metrics and tracking_metrics to ax
            raw_data = {}
            for j in range(len(self.all_metrics)):
                raw_data[self.all_metrics[j]] = fac*Y_turbo[i][j].item()
            if Y_tracking is not None:
                for j in range(len(self.all_tracking_metrics)):
                    raw_data[self.all_tracking_metrics[j]] = Y_tracking[i][j].item()
            self.ax_client.complete_trial(trial_index, raw_data=raw_data)

        # train the model
        self.ax_client.get_next_trial(1) # This will train the model

        if verbose_logging:
            logging_level = 20
            logger.setLevel(logging_level)
            if state.restart_triggered:
                logger.info('Turbo converged after %d batches with %d trials', count_batch-1, (count_batch-1)*state.batch_size)
            else:
                logger.info('Turbo is terminated as the max number (%d) of trials is reached', total_trials)
        # if verbose_logging:
        #     logging_level = 20
        #     logger.setLevel(logging_level)
        #     logger.info('Finished Turbo')

    def update_params_with_best_balance(self,return_best_balance=False):
        """ Update the parameters with the best balance of all metrics. 
        The best balance is defined by ranking the results for each metric and taking the parameters that has the lowest sum of ranks.
        
        Raises
        ------
        ValueError
            We need at least one metric to update the parameters
        """        

        # if we have one objective
        if len(self.all_metrics) == 1:
            scaled_best_parameters = self.ax_client.get_best_parameters()[0]
            self.params_w(scaled_best_parameters,self.params)
        # if we have multiple objectives
        elif len(self.all_metrics) > 1:
            # We do this because the ax_client.get_pareto_optimal_parameters does not necessarily return the best parameters for a balanced results on all objectives
            df = get_df_ax_client_metrics(self.params, self.ax_client, self.all_metrics)
            metrics = self.all_metrics
            minimizes_ = []

            for agent in self.agents:
                for i in range(len(agent.minimize)):
                    minimizes_.append(agent.minimize[i])

            ranked_df = copy.deepcopy(df)
            ranks = []
            for i in range(len(metrics)):
                ranked_df[metrics[i]+'_rank'] = ranked_df[metrics[i]].rank(ascending=minimizes_[i])
                ranks.append(ranked_df[metrics[i]+'_rank'])
            # get the index of the best balance
            best_balance_index = np.argmin(np.sum(np.array(ranks), axis=0))

            # get the best parameters
            scaled_best_parameters = ranked_df.iloc[best_balance_index].to_dict()
            
            dum_dic = {}
            for p in self.params:
                dum_dic[p.name] = scaled_best_parameters[p.name]
            scaled_best_parameters = dum_dic

            for p in self.params:
                if p.name in scaled_best_parameters.keys():
                    p.value = scaled_best_parameters[p.name]
            if return_best_balance:
                return best_balance_index, scaled_best_parameters
        else:
            raise ValueError('We need at least one metric to update the parameters')

              
######### Turbo specific functions ##############################################################
@dataclass
class TurboState:
    dim: int
    batch_size: int
    length: float = 0.8
    length_min: float = 0.5**7
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = float("nan")  # Note: Post-initialized
    success_counter: int = 0
    success_tolerance: int = 3  # Note: The original paper uses 3
    best_value: float = -float("inf")
    restart_triggered: bool = False
    def __init__(self, dim, batch_size, best_value, **kwargs):
        self.dim = dim
        self.batch_size = batch_size
        self.best_value = best_value
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.failure_tolerance = math.ceil(
                max([4.0 / self.batch_size, float(self.dim) / self.batch_size])
            )
        
    # def __post_init__(self):
        

def get_initial_points(dim, n_pts, seed=None, device=None, dtype=None):
    """ Generate initial points using Sobol sequence.

    Parameters
    ----------
    dim : int
        Number of dimensions
    n_pts : int
        Number of points to generate
    seed : int, optional
        Random seed, by default None
    device : torch.device, optional
        Device to use for the generated points, by default None
    dtype : torch.dtype, optional
        Data type of the generated points, by default None

    Returns
    -------
    torch.Tensor
        Generated points in the range [0, 1]^d
    """    
    sobol = SobolEngine(dimension=dim, scramble=True, seed=seed)
    X_init = sobol.draw(n_pts).to(dtype=dtype, device=device)
    return X_init

def update_state(state, Y_next):
    """ Update the state of the optimization process based on the new observations.
       For TURBO optimization only.
       The state is updated based on the success or failure of the new observations.

    Parameters
    ----------
    state : TurboState
        Current state of the optimization process
    Y_next : torch.Tensor
        New observations

    Returns
    -------
    TurboState
        Updated state of the optimization process

    """    
    # For maximization, we want the maximum value
    current_best = max(Y_next).item()
    is_success = current_best > state.best_value + 1e-3 * math.fabs(state.best_value)
    state.best_value = max(state.best_value, current_best)
    
    if is_success:
        state.success_counter += 1
        state.failure_counter = 0
    else:
        state.success_counter = 0
        state.failure_counter += 1

    if state.success_counter == state.success_tolerance:  # Expand trust region
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter == state.failure_tolerance:  # Shrink trust region
        state.length /= 2.0
        state.failure_counter = 0

    if state.length < state.length_min:
        state.restart_triggered = True
    return state

def generate_batch(state, model, X,  # Evaluated points on the domain [0, 1]^d
    Y,  # Function values
    batch_size,
    n_candidates=None,  # Number of candidates for Thompson sampling
    num_restarts=10,
    raw_samples=512,
    acqf="ts",  # "ei" or "ts"
    device=None,
    dtype=None,
    minimize=False,
):
    """ Generate a batch of points using the TURBO algorithm.
    The batch is generated using either Thompson sampling or Expected Improvement.

    Parameters
    ----------
    state : TurboState
        Current state of the optimization process
    model : GPyTorchModel
        GPyTorch model for the function
    X : torch.Tensor
        Evaluated points on the domain [0, 1]^d
    Y : torch.Tensor
        Function values
    batch_size : int
        Number of points to generate
    n_candidates : int, optional
        Number of candidates for Thompson sampling, by default None
    num_restarts : int, optional
        Number of restarts for the optimization, by default 10
    raw_samples : int, optional
        Number of raw samples for the optimization, by default 512
    acqf : str, optional
        Acquisition function to use can be "ts" or "ei", by default "ts"
    device : torch.device, optional
        Device to use for the generated points, by default None
    dtype : torch.dtype, optional
        Data type of the generated points, by default None
    minimize : bool, optional
        Whether to minimize or maximize the function, by default False

    Returns
    -------
    torch.Tensor
        Generated points in the range [0, 1]^d
    
    Raises
    ------
    AssertionError
        If the acquisition function is not "ts" or "ei"
    ValueError
        If the acquisition function is not "ts" or "ei"
    """    
    assert acqf in ("ts", "ei")
    assert X.min() >= 0.0 and X.max() <= 1.0 and torch.all(torch.isfinite(Y))
    if n_candidates is None:
        n_candidates = min(5000, max(2000, 200 * X.shape[-1]))

    # Scale the TR to be proportional to the lengthscales
    x_center = X[Y.argmax(), :].clone()
        
    weights = model.covar_module.base_kernel.lengthscale.squeeze().detach()
    weights = weights / weights.mean()
    weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
    tr_lb = torch.clamp(x_center - weights * state.length / 2.0, 0.0, 1.0)
    tr_ub = torch.clamp(x_center + weights * state.length / 2.0, 0.0, 1.0)

    if acqf == "ts":
        # Thompson Sampling
        dim = X.shape[-1]
        sobol = SobolEngine(dim, scramble=True, )#seed=np.random.randint(10000))
        pert = sobol.draw(n_candidates).to(dtype=dtype, device=device)
        pert = tr_lb + (tr_ub - tr_lb) * pert

        # Create a perturbation mask
        prob_perturb = min(20.0 / dim, 1.0)
        mask = torch.rand(n_candidates, dim, dtype=dtype, device=device) <= prob_perturb
        ind = torch.where(mask.sum(dim=1) == 0)[0]
        mask[ind, torch.randint(0, dim - 1, size=(len(ind),), device=device)] = 1

        # Create candidate points from the perturbations and the mask
        X_cand = x_center.expand(n_candidates, dim).clone()
        X_cand[mask] = pert[mask]

        # Sample from the posterior
        thompson_sampling = MaxPosteriorSampling(model=model, replacement=False)
            
        with torch.no_grad():
            X_next = thompson_sampling(X_cand, num_samples=batch_size)

    elif acqf == "ei":
        # Expected Improvement
        from botorch.optim import optimize_acqf
        if minimize:
            best_f = Y.min().item()
        else:
            best_f = Y.max().item()
        
        # Use qLogExpectedImprovement for better numerical stability
        acq_func = qLogExpectedImprovement(
            model=model,
            best_f=best_f,
        )#
        
        X_next, acq_value = optimize_acqf(
            acq_function=acq_func,
            bounds=torch.stack([tr_lb, tr_ub]),
            q=batch_size,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            options={"batch_limit": 5, "maxiter": 200},
        )
    else:
        raise ValueError(f"Unknown acquisition function type: {acqf}")

    return X_next
if __name__ == '__main__':
    pass