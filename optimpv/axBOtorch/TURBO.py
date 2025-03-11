#!/usr/bin/env python3



# DO NOT USE IT IS NOT WORKING!!!!!!!!!!!!!!!!



from collections.abc import Callable
from dataclasses import dataclass
import math
from copy import deepcopy
from functools import partial
from logging import Logger
from typing import Any, Optional, Dict, Tuple, List
import numpy as np
import os
import pickle

import torch
from torch import Tensor
from torch.quasirandom import SobolEngine
from ax.core.search_space import SearchSpaceDigest
from ax.models.torch.botorch_modular.acquisition import Acquisition
from ax.models.torch.botorch_modular.surrogate import Surrogate
from ax.models.torch_base import TorchOptConfig
from ax.utils.common.logger import get_logger
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.generation import MaxPosteriorSampling
from botorch.models.model import Model
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.utils.transforms import unnormalize, normalize
from botorch.acquisition.logei import (
    qLogExpectedImprovement,
    qLogNoisyExpectedImprovement,
)

CLAMP_TOL = 1e-2
logger: Logger = get_logger(__name__)

@dataclass
class TurboState:
    dim: int = 1
    batch_size: int = 1
    length: float = 0.8
    length_min: float = 1e-6 #0.5**7
    length_max: float = 100 #1.6
    failure_counter: int = 0
    failure_tolerance: int = 0  # Will be set in post_init
    success_counter: int = 0
    success_tolerance: int = 3 #10  # The original paper uses 3
    best_value: float = -float("inf")
    restart_triggered: bool = False

    def __post_init__(self):
        self.failure_tolerance = math.ceil(
            max([4.0 / self.batch_size, float(self.dim) / self.batch_size])
        )


def update_state(state: TurboState, Y_next: Tensor, minimize: bool = False) -> TurboState:
    """Update the TurboState based on new function evaluations.

    Args:
        state: The current TurboState.
        Y_next: The new function values.
        minimize: Whether to minimize the function.

    Returns:
        The updated TurboState.
    """
    if minimize:
        # For minimization problems
        if torch.min(Y_next) < state.best_value - 1e-3 * math.fabs(state.best_value):
            state.success_counter += 1
            state.failure_counter = 0
        else:
            state.success_counter = 0
            state.failure_counter += 1
            
        # Update best value (minimum for minimization)
        state.best_value = min(state.best_value, torch.min(Y_next).item())
    else:
        # Original code for maximization
        if torch.max(Y_next) > state.best_value + 1e-3 * math.fabs(state.best_value):
            state.success_counter += 1
            state.failure_counter = 0
        else:
            state.success_counter = 0
            state.failure_counter += 1
            
        # Update best value (maximum for maximization)
        state.best_value = max(state.best_value, torch.max(Y_next).item())

    # Update trust region size based on counters
    if state.success_counter == state.success_tolerance:  # Expand trust region
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter == state.failure_tolerance:  # Shrink trust region
        state.length /= 2.0
        state.failure_counter = 0

    # Check if restart is needed
    if state.length < state.length_min:
        state.restart_triggered = True
    
    return state


class TURBOAcquisition(Acquisition):
    """
    Implement Trust Region Bayesian Optimization (TuRBO) acquisition function.
    
    Based on the paper:
    Eriksson, D., Pearce, M., Gardner, J., Turner, R. D., & Poloczek, M. (2019).
    Scalable global optimization via local Bayesian optimization.
    Advances in Neural Information Processing Systems.
    """

    def __init__(
        self,
        surrogate: Surrogate,
        search_space_digest: SearchSpaceDigest,
        torch_opt_config: TorchOptConfig,
        botorch_acqf_class: Optional[type[AcquisitionFunction]] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize TuRBO acquisition function.

        Args:
            surrogate: The surrogate model.
            search_space_digest: Search space digest.
            torch_opt_config: Torch optimization config.
            botorch_acqf_class: Not used for TuRBO, as it uses Thompson Sampling or EI.
            options: Additional options with the following keys:
                - acqf_type: Acquisition function type, either "ts" (Thompson Sampling)
                  or "ei" (Expected Improvement). Default is "ts".
                - n_candidates: Number of candidates for TS. Default is min(5000, max(2000, 200*dim)).
        """
        tkwargs: Dict[str, Any] = {"dtype": surrogate.dtype, "device": surrogate.device}
        options = {} if options is None else options
        surrogate_f = deepcopy(surrogate)

        # Initialize TuRBO state
        self.batch_size = torch_opt_config.model_gen_options.get("batch_size", 1)
        print(self.batch_size)
        # Get dimension safely, handling both list and tensor bounds
        if isinstance(search_space_digest.bounds, list):
            dim = len(search_space_digest.bounds[0])  # Number of parameters
        else:
            dim = search_space_digest.bounds.shape[0]
        
        
        # Store TuRBO-specific options separately
        self.turbo_options = {
            "acqf_type": options.get("acqf_type", "ts"),
            "n_candidates": options.get("n_candidates", None),
            "raw_samples": options.get("raw_samples", 512),
            "num_restarts": options.get("num_restarts", 10),
            "state_file": options.get("state_file", None),
            "save_on_update": options.get("save_on_update", False),
            "minimize": options.get("minimize", False),
        }
        self.minimize = self.turbo_options.get("minimize", False)
        print(self.turbo_options.get("minimize", False))
        # Initialize state as an instance variable
        if self.turbo_options.get("minimize", False):
            print("minimizing")
            self.state = TurboState(dim=dim, batch_size=self.batch_size, best_value=float("inf"))
            print(self.state)
        else:
            self.state = TurboState(dim=dim, batch_size=self.batch_size)
        
        print(f"TuRBO options: {self.turbo_options}")
        # Load state from file if specified
        state_file = self.turbo_options.get("state_file")
        if state_file:
            self.load_state(state_file)
        
        # Remove TuRBO-specific options from what gets passed to the parent
        parent_options = {k: v for k, v in options.items() 
                         if k not in ["acqf_type", "n_candidates", "state_file", 
                                     "save_on_update", "raw_samples", "num_restarts","minimize"]}
        # remove them from the options passed to the parent class
        # self.options = parent_options
        # Track X and Y history for TuRBO's local optimization
        self.X_history = None
        self.Y_history = None

        # Pass qLogNoisyExpectedImprovement as the acquisition function class instead of None
        # This prevents issubclass() errors in Ax's internal code
        # remove turbo options from the options passed to the parent class
        # parent_options = {k: v for k, v in options.items()
        #                   if k not in ["acqf_type", "n_candidates", "state_file", "save_on_update"]}
        super().__init__(
            surrogate=surrogate_f,
            search_space_digest=search_space_digest,
            torch_opt_config=torch_opt_config,
            botorch_acqf_class=qLogNoisyExpectedImprovement,  # Use a concrete class instead of None
            options=parent_options,  # Use filtered options
        )

    def save_state(self, filepath: str) -> None:
        """Save the current TuRBO state to a file."""
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            
        with open(filepath, 'wb') as f:
            pickle.dump(self.state, f)
        logger.info(f"Saved TuRBO state to {filepath}")

    def load_state(self, filepath: str) -> bool:
        """Load TuRBO state from a file."""
        if os.path.exists(filepath):
            try:
                with open(filepath, 'rb') as f:
                    self.state = pickle.load(f)
                logger.info(f"Loaded TuRBO state from {filepath}: best_value={self.state.best_value:.4f}, length={self.state.length:.4f}")
                return True
            except Exception as e:
                logger.error(f"Error loading TuRBO state: {e}")
                return False
        return False

    def update_state_from_Y(self, Y_next: Tensor) -> None:
        """Update the TuRBO state based on new function evaluations.

        Args:
            Y_next: The new function values.
        """
        self.state = update_state(state=self.state, Y_next=Y_next)
        print(f"Updated TuRBO state: Best value = {self.state.best_value:.4f}, ")
        # Save state if requested
        if self.turbo_options.get("save_on_update") and self.turbo_options.get("state_file"):
            self.save_state(self.turbo_options["state_file"])

    def update_history(self, X: Tensor, Y: Tensor) -> None:
        """Update the history of evaluated points.

        Args:
            X: New points.
            Y: New function values.
        """
        if self.X_history is None:
            self.X_history = X
            self.Y_history = Y
        else:
            self.X_history = torch.cat([self.X_history, X], dim=0)
            self.Y_history = torch.cat([self.Y_history, Y], dim=0)

    def generate_batch(
        self,
        model: Model,
        X: Tensor,
        Y: Tensor,
        bounds: Tensor,
        num_restarts: int = 10,
        raw_samples: int = 512,
        acqf_type: str = "ts",
        n_candidates: Optional[int] = None,
        minimize: bool = False,
    ) -> Tensor:
        """Generate a batch of candidates within the trust region.

        Args:
            model: The surrogate model.
            X: The normalized points evaluated so far.
            Y: The function values.
            bounds: The normalized bounds [2, dim].
            num_restarts: Number of restarts for EI optimization.
            raw_samples: Number of raw samples for EI optimization.
            acqf_type: Acquisition function type, either "ts" or "ei".
            n_candidates: Number of candidates for TS.

        Returns:
            A batch of candidates.
        """
        device = X.device
        dtype = X.dtype
        dim = X.shape[-1]

        if n_candidates is None:
            n_candidates = min(5000, max(2000, 200 * dim))

        # Scale the TR to be proportional to the lengthscales
        if self.minimize:
            x_center = X[Y.argmin(), :].clone()
        else:
            x_center = X[Y.argmax(), :].clone()
        
        # Get lengthscales from the model
        if isinstance(model, ModelListGP):
            # For multi-output models
            try:
                # Try to access base_kernel (for RBFKernel and similar)
                lengthscales = model.models[0].covar_module.base_kernel.lengthscale.detach()
            except:
                # Fall back to direct lengthscale (for MaternKernel)
                lengthscales = model.models[0].covar_module.lengthscale.detach()

            if lengthscales.ndim > 1:
                lengthscales = lengthscales.squeeze(0) 
        else:
            # For single-output models
            try:
                lengthscales = model.covar_module.base_kernel.lengthscale.detach()
            except:
                lengthscales = model.covar_module.lengthscale.detach()
                
            if lengthscales.ndim > 1:
                lengthscales = lengthscales.squeeze(0)

        weights = lengthscales / lengthscales.mean()
        weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
        
        # Set the trust region bounds
        tr_lb = torch.clamp(x_center - weights * self.state.length / 2.0, 0.0, 1.0)
        tr_ub = torch.clamp(x_center + weights * self.state.length / 2.0, 0.0, 1.0)

        if acqf_type == "ts":
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
            if minimize:
                # For minimization: use a negative objective
                from botorch.acquisition.objective import ScalarizedObjective
                weights = torch.tensor([-1.0], device=device, dtype=dtype)
                obj = ScalarizedObjective(weights=weights)
                thompson_sampling = MaxPosteriorSampling(model=model, replacement=False, objective=obj)
            else:
                # Original code for maximization
                thompson_sampling = MaxPosteriorSampling(model=model, replacement=False)
                
            with torch.no_grad():
                X_next = thompson_sampling(X_cand, num_samples=self.batch_size)

        elif acqf_type == "ei":
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
                q=self.batch_size,
                num_restarts=num_restarts,
                raw_samples=raw_samples,
                options={"batch_limit": 5, "maxiter": 200},
            )
        else:
            raise ValueError(f"Unknown acquisition function type: {acqf_type}")

        return X_next

    def optimize(
        self,
        n: int,
        search_space_digest: SearchSpaceDigest,
        inequality_constraints: Optional[List[Tuple[Tensor, Tensor, float]]] = None,
        fixed_features: Optional[Dict[int, float]] = None,
        rounding_func: Optional[Callable[[Tensor], Tensor]] = None,
        optimizer_options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Generate a set of candidates via TuRBO.

        Args:
            n: The number of candidates to generate.
            search_space_digest: A SearchSpaceDigest object.
            inequality_constraints: Inequality constraints.
            fixed_features: Fixed features.
            rounding_func: A function for rounding.
            optimizer_options: Options for the optimizer.

        Returns:
            A three-element tuple containing:
            - an n x d tensor of generated candidates
            - a tensor with the associated acquisition values
            - a tensor with weights for each candidate
        """
        device = self.device
        dtype = self.dtype
        tkwargs = {"device": device, "dtype": dtype}
        bounds = torch.tensor(search_space_digest.bounds, **tkwargs).t()
        self.batch_size = n
        
        # Get model and data
        model = self.surrogate.model
        
        # Extract training data from the surrogate
        X, Y = [], []
        for dataset in self.surrogate.training_data:
            X.append(dataset.X)
            Y.append(dataset.Y)
        
        X = torch.cat(X, dim=0)  # [batch_size, dim]
        Y = torch.cat(Y, dim=0)  # [batch_size, 1]
        
        if X.shape[0] == 0 or Y.shape[0] == 0:
            raise ValueError("No training data available in the surrogate model.")
        else:
            # load the state from the file if it exists
            if self.turbo_options.get("state_file"):
                self.load_state(self.turbo_options["state_file"])
            print(f"TuRBO state: {self.state}")
            # update the state based on the training data
            self.state = update_state(state=self.state, Y_next=Y,minimize=self.turbo_options.get("minimize", False))
            # save the state if requested
            if self.turbo_options.get("save_on_update") and self.turbo_options.get("state_file"):
                self.save_state(self.turbo_options["state_file"])

        print(
            f"{len(X)}) Best value: {self.state.best_value:.2e}, TR length: {self.state.length:.2e}"
        )

        # Generate candidates using TuRBO's trust region
        acqf_type = self.turbo_options.get("acqf_type", "ts")
        n_candidates = self.turbo_options.get("n_candidates", None)
        raw_samples = self.turbo_options.get("raw_samples", 512)
        num_restarts = self.turbo_options.get("num_restarts", 10)
        
        X_next = self.generate_batch(
            model=model,
            X=X,
            Y=Y,
            bounds=bounds,
            acqf_type=acqf_type,
            n_candidates=n_candidates,
            raw_samples=raw_samples,
            num_restarts=num_restarts,
        )

        candidates = X_next

        # For TuRBO, we don't have explicit acquisition values
        # So we use the model to predict the values
        with torch.no_grad():
            # For multi-objective models, we sum the predictions
            if isinstance(model, ModelListGP):
                acquisition_value = torch.zeros(len(candidates), device=device, dtype=dtype)
                for i in range(len(model.models)):
                    pred = model.models[i](candidates).mean
                    acquisition_value += pred.squeeze(-1)
            else:
                acquisition_value = model(candidates).mean.squeeze(-1)


        
        return candidates, acquisition_value, torch.ones(n, device=candidates.device, dtype=candidates.dtype)

    def evaluate(self, X: Tensor) -> Tensor:
        """Evaluate the acquisition function on the given candidates.

        Args:
            X: A batch_shape x q x d-dim tensor of candidates.

        Returns:
            A batch_shape-dim tensor of acquisition values.
        """
        # For TuRBO, we don't have a standard acquisition function
        # but we can use the model to predict values
        with torch.no_grad():
            # For multi-objective models, we sum the predictions
            if isinstance(self.surrogate.model, ModelListGP):
                acq_values = torch.zeros(
                    X.shape[0], device=X.device, dtype=X.dtype
                )
                for i in range(len(self.surrogate.model.models)):
                    pred = self.surrogate.model.models[i](X).mean
                    acq_values += pred.squeeze(-1)
            else:
                acq_values = self.surrogate.model(X).mean.squeeze(-1)
                
        return acq_values
