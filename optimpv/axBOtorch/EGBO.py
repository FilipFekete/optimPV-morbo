#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

# import functools
# import warnings
from collections.abc import Callable
from copy import deepcopy
from functools import partial
from logging import Logger
from typing import Any

import torch
from ax.core.search_space import SearchSpaceDigest
# from ax.exceptions.core import AxWarning
from ax.models.torch.botorch_modular.acquisition import Acquisition
# from ax.models.torch.botorch_modular.optimizer_argparse import optimizer_argparse
from ax.models.torch.botorch_modular.surrogate import Surrogate
from ax.models.torch_base import TorchOptConfig
from ax.utils.common.logger import get_logger
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.multi_objective.logei import (
    qLogNoisyExpectedHypervolumeImprovement,
)
# from botorch.acquisition.penalized import L0Approximation
# from botorch.models.deterministic import GenericDeterministicModel
from botorch.models.model import ModelList
# from botorch.optim import (
#     gen_batch_initial_conditions,
#     Homotopy,
#     HomotopyParameter,
#     LogLinearHomotopySchedule,
#     optimize_acqf_homotopy,
# )
# from botorch.utils.datasets import SupervisedDataset
# from pyre_extensions import assert_is_instance, none_throws
from torch import Tensor

CLAMP_TOL = 1e-2
logger: Logger = get_logger(__name__)

# added Vincent
# from botorch.utils.transforms import unnormalize, normalize
# from botorch.utils.constraints import get_outcome_constraint_transforms
from torch import Tensor
# from ax.models.model_utils import get_observed
# from botorch.optim.optimize import optimize_acqf, optimize_acqf_list
# from botorch.utils.multi_objective.pareto import is_non_dominated
# from ax.utils.common.constants import Keys
from botorch.utils.multi_objective.hypervolume import Hypervolume
# from botorch.optim.optimize import optimize_acqf, optimize_acqf_list
from botorch.utils.multi_objective.pareto import is_non_dominated

import pymoo
# from pymoo.core.problem import ElementwiseProblem
# from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.util.ref_dirs import get_reference_directions
# from pymoo.optimize import minimize
from pymoo.core.problem import Problem 
# from pymoo.core.population import Population
from pymoo.core.termination import NoTermination

# from pymoo.core.individual import Individual
# from pymoo.operators.crossover.sbx import SBX
# from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.util.ref_dirs import get_reference_directions
# from sklearn.preprocessing import MinMaxScaler


# from pymoo.problems import get_problem
# from pymoo.core.problem import ElementwiseProblem

from pymoo.algorithms.moo.unsga3 import UNSGA3
from pymoo.util.ref_dirs import get_reference_directions
# from pymoo.optimize import minimize

# from pymoo.core.problem import Problem as PymooProblem
from pymoo.core.termination import NoTermination
# from botorch.utils.sampling import draw_sobol_samples
import numpy as np

class EGBOAcquisition(Acquisition):
    """
    Implement the acquisition function of Evolution-Guided Bayesian Optimization (EGBO).  

    Based on the following paper:  
    Low, A.K.Y., Mekki-Berrada, F., Gupta, A. et al. Evolution-guided Bayesian optimization for constrained multi-objective optimization in self-driving labs. npj Comput Mater 10, 104 (2024). https://doi.org/10.1038/s41524-024-01274-x

    """

    def __init__(
        self,
        surrogate: Surrogate,
        search_space_digest: SearchSpaceDigest,
        torch_opt_config: TorchOptConfig,
        botorch_acqf_class: type[AcquisitionFunction],
        options: dict[str, Any] | None = None,
    ) -> None:
        tkwargs: dict[str, Any] = {"dtype": surrogate.dtype, "device": surrogate.device}
        options = {} if options is None else options
        surrogate_f = deepcopy(surrogate)
        surrogate_f._model = ModelList(surrogate.model) #

        super().__init__(
            surrogate=surrogate_f,
            search_space_digest=search_space_digest,
            torch_opt_config=torch_opt_config,
            botorch_acqf_class=qLogNoisyExpectedHypervolumeImprovement,
            options=options,
        )


    def optimize_(
        self,
        n: int,
        search_space_digest: SearchSpaceDigest,
        inequality_constraints: list[tuple[Tensor, Tensor, float]] | None = None,
        fixed_features: dict[int, float] | None = None,
        rounding_func: Callable[[Tensor], Tensor] | None = None,
        optimizer_options: dict[str, Any] | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Generate a set of candidates via multi-start optimization. Obtains
        candidates and their associated acquisition function values.

        Args:
            n: The number of candidates to generate.
            search_space_digest: A ``SearchSpaceDigest`` object containing search space
                properties, e.g. ``bounds`` for optimization.
            inequality_constraints: A list of tuples (indices, coefficients, rhs),
                with each tuple encoding an inequality constraint of the form
                ``sum_i (X[indices[i]] * coefficients[i]) >= rhs``.
            fixed_features: A map `{feature_index: value}` for features that
                should be fixed to a particular value during generation.
            rounding_func: A function that post-processes an optimization
                result appropriately (i.e., according to `round-trip`
                transformations).
            optimizer_options: Options for the optimizer function, e.g. ``sequential``
                or ``raw_samples``.

        Returns:
            A three-element tuple containing an `n x d`-dim tensor of generated
            candidates, a tensor with the associated acquisition values, and a tensor
            with the weight for each candidate.
        """
        # if self.penalty_name == "L0_norm":
        #     candidates, expected_acquisition_value, weights = (
        #         self._optimize_with_homotopy(
        #             n=n,
        #             search_space_digest=search_space_digest,
        #             inequality_constraints=inequality_constraints,
        #             fixed_features=fixed_features,
        #             rounding_func=rounding_func,
        #             optimizer_options=optimizer_options,
        #         )
        #     )
        # else:
        # if L1 norm use standard moo-opt
        candidates, expected_acquisition_value, weights = super().optimize(
            n=n,
            search_space_digest=search_space_digest,
            inequality_constraints=inequality_constraints,
            fixed_features=fixed_features,
            rounding_func=rounding_func,
            optimizer_options=optimizer_options,
        )

        # similar, make sure if applies to sparse dimensions only
        # candidates = clamp_to_target(
        #     X=candidates, target_point=self.target_point, clamp_tol=CLAMP_TOL
        # )
        return candidates, expected_acquisition_value, weights
    
    def optimize(
        self,
        n: int,
        search_space_digest: SearchSpaceDigest,
        inequality_constraints: list[tuple[Tensor, Tensor, float]] | None = None,
        fixed_features: dict[int, float] | None = None,
        rounding_func: Callable[[Tensor], Tensor] | None = None,
        optimizer_options: dict[str, Any] | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Generate a set of candidates via multi-start optimization. Obtains
        candidates and their associated acquisition function values.

        Args:
            n: The number of candidates to generate.
            search_space_digest: A ``SearchSpaceDigest`` object containing search space
                properties, e.g. ``bounds`` for optimization.
            inequality_constraints: A list of tuples (indices, coefficients, rhs),
                with each tuple encoding an inequality constraint of the form
                ``sum_i (X[indices[i]] * coefficients[i]) >= rhs``.
            fixed_features: A map `{feature_index: value}` for features that
                should be fixed to a particular value during generation.
            rounding_func: A function that post-processes an optimization
                result appropriately (i.e., according to `round-trip`
                transformations).
            optimizer_options: Options for the optimizer function, e.g. ``sequential``
                or ``raw_samples``.

        Returns:
            A three-element tuple containing an `n x d`-dim tensor of generated
            candidates, a tensor with the associated acquisition values, and a tensor
            with the weight for each candidate.
        """
        # Get the device information and the optimizer options
        device = self.device
        dtype = self.dtype
        tkwargs = {"device": device, "dtype": dtype}
        acq_func = self.acqf
        _tensorize = partial(torch.tensor, dtype=self.dtype, device=self.device)
        ssd = search_space_digest
        bounds = _tensorize(ssd.bounds).t()

        # First, we optimize the acquisition function using the standard method
        candidates, expected_acquisition_value, weights = self.optimize_(
            n=n,
            search_space_digest=search_space_digest,
            inequality_constraints=inequality_constraints,
            fixed_features=fixed_features,
            rounding_func=rounding_func,
            optimizer_options=optimizer_options,
        )
        qnehvi_x = candidates

        # Next, we optimize the acquisition function using NSGA3
        # get the training data
        Xs = []
        Ys = []
        for dataset in self.surrogate.training_data:
            Xs.append(dataset.X)
            Ys.append(dataset.Y)
        x = Xs[0]
        y = Ys[0]
        n_obj = y.shape[1] # number of objectives
        n_var = x.shape[1] # number of variables
        n_constr = 0 # number of constraints

        # we pick out the best points so far to form parents
        pareto_mask = is_non_dominated(y)
        pareto_x = x[pareto_mask].cpu().numpy()
        pareto_y = -y[pareto_mask].cpu().numpy()

        hv=Hypervolume(ref_point=-self.acqf.ref_point)
        algorithm = UNSGA3(pop_size=256,
                           ref_dirs=get_reference_directions("energy", n_obj, n, seed=None),
                           sampling=pareto_x,
                        #    sampling = qnehvi_x.cpu().numpy(),
                           #crossover=SimulatedBinaryCrossover(eta=30, prob=1.0),
                           #mutation=PolynomialMutation(eta=20, prob=None),
                          )
        
        # make xl, xu from the bounds
        xl = bounds[0].cpu().numpy()
        xu = bounds[1].cpu().numpy()

        # define the pymoo problem
        pymooproblem = Problem(n_var=n_var, n_obj=n_obj, n_constr=n_constr, 
                      xl=xl, xu=xu)

        algorithm.setup(pymooproblem, termination=NoTermination())
        
        # set the 1st population to the current evaluated population
        pop = algorithm.ask()
        pop.set("F", pareto_y)
        # pop.set("G", pareto_y)
        algorithm.tell(infills=pop)

        # propose children based on tournament selection -> crossover/mutation
        newpop = algorithm.ask()
        nsga3_x = torch.tensor(newpop.get("X"), **tkwargs)
        
        # total pool of candidates for sorting
        candidates = torch.cat([qnehvi_x, nsga3_x])
        acq_value_list = []
        for i in range(0, candidates.shape[0]):
            with torch.no_grad():
                acq_value = acq_func(candidates[i].unsqueeze(dim=0))
                acq_value_list.append(acq_value.item())

        sorted_x = candidates.cpu().numpy()[np.argsort(acq_value_list)]
        
        acqf_values = torch.tensor(acq_value_list, **tkwargs)
        candidates = torch.tensor(sorted_x[-n:], **tkwargs)
        acqf_values = acqf_values[-n:]
        expected_acquisition_value = acqf_values

        return candidates, expected_acquisition_value, weights
