"""Trust Region Bayesian Optimization (TuRBO) generation node for us in Ax."""

# pyright: basic

import math
from typing import Any, Literal, Self, override

import torch
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.parameter import RangeParameter
from ax.core.trial_status import TrialStatus
from ax.core.types import TParameterization
from ax.exceptions.core import OptimizationShouldStop
from ax.generation_strategy.external_generation_node import ExternalGenerationNode
from botorch.acquisition import qLogExpectedImprovement
from botorch.fit import fit_gpytorch_mll  # pyright: ignore[reportUnknownVariableType]
from botorch.generation import MaxPosteriorSampling
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf  # pyright: ignore[reportUnknownVariableType]
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from pydantic import BaseModel
from torch.quasirandom import SobolEngine


def fit_gp(
    X: torch.Tensor,
    Y: torch.Tensor,
    *,
    normalize_inputs: bool = True,
    standardize_outputs: bool = True,
    max_cholesky_size: float = float("inf"),  # pyright: ignore[reportCallInDefaultInitializer]
) -> SingleTaskGP:
    """Build and fit a GP model using BoTorch defaults.

    Args:
        X: Training inputs of shape (n, d).
        Y: Training targets of shape (n, 1).
        normalize_inputs: Whether to normalize inputs to [0, 1]^d.
        standardize_outputs: Whether to standardize outputs to zero mean, unit variance.
        max_cholesky_size: Max size for Cholesky decomposition.

    Returns:
        Fitted GP model.
    """
    import gpytorch

    d = X.shape[-1]

    input_transform = Normalize(d=d) if normalize_inputs else None
    outcome_transform = Standardize(m=1) if standardize_outputs else None

    covar_module = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=d))

    model = SingleTaskGP(
        X,
        Y,
        covar_module=covar_module,
        input_transform=input_transform,
        outcome_transform=outcome_transform,
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)

    with gpytorch.settings.max_cholesky_size(max_cholesky_size):
        fit_gpytorch_mll(mll)  # pyright: ignore[reportUnusedCallResult]

    return model

class TurboConvergedError(OptimizationShouldStop):
    """Raised when TuRBO has converged (trust region below minimum)."""

    pass


class TurboState(BaseModel):
    """State of the TuRBO algorithm."""

    dim: int
    batch_size: int
    length: float = 0.8
    length_min: float = 0.5**7
    length_max: float = 1.6
    failure_counter: int = 0
    success_counter: int = 0
    success_tolerance: int = 10
    failure_tolerance: float | None = None
    best_value: float | None = None
    restart_triggered: bool = False
    maximize: bool = True  # Whether we are maximizing or minimizing the objective

    def model_post_init(self, __context: Any) -> None:  # noqa: PYI063
        """Initialize optional fields based on the required fields."""
        if self.best_value is None:
            self.best_value = -float("inf") if self.maximize else float("inf")

        if self.failure_tolerance is None:
            self.failure_tolerance = math.ceil(max([4.0 / self.batch_size, float(self.dim) / self.batch_size]))

    def update_state(self, Y_next: torch.Tensor) -> Self:
        """Update the state of the TuRBO algorithm. If the new set of points
        has an improvement over the best value, we increase the success counter.
        If the new set of points does not have an improvement, we increase the
        failure counter. The size of the trust region is updated based on the
        running values of the success and failure counters. If the trust region
        becomes too small then we trigger a restart.
        """
        if self.best_value is None:
            raise RuntimeError("Best value not initialized. This is a bug.")

        if self.maximize:
            best_new = Y_next.max().item()
            is_improvement = best_new > self.best_value + 1e-3 * math.fabs(self.best_value)  # pyright: ignore[reportArgumentType, reportOptionalOperand]
        else:
            best_new = Y_next.min().item()
            is_improvement = best_new < self.best_value - 1e-3 * math.fabs(self.best_value)  # pyright: ignore[reportArgumentType, reportOptionalOperand]

        if is_improvement:
            self.success_counter += 1
            self.failure_counter = 0
        else:
            self.success_counter = 0
            self.failure_counter += 1

        if self.success_counter == self.success_tolerance:  # Expand trust region
            self.length = min(2.0 * self.length, self.length_max)
            self.success_counter = 0
        elif self.failure_counter == self.failure_tolerance:  # Shrink trust region
            self.length /= 2.0
            self.failure_counter = 0

        if self.maximize:
            self.best_value = max(self.best_value, best_new)
        else:
            self.best_value = min(self.best_value, best_new)

        if self.length < self.length_min:
            self.restart_triggered = True
        return self

    def get_trust_region_bounds(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        *,
        buffer: float = 0.0,
        weights: torch.Tensor | None = None,
        maximize: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get the trust region bounds for the TuRBO algorithm."""
        maximize_mode = self.maximize if maximize is None else maximize
        x_center = X[Y.argmax() if maximize_mode else Y.argmin(), :].clone()
        if weights is None:
            weights = torch.ones_like(x_center)  # Initial weights before model fitting
        else:
            if weights.shape != x_center.shape:
                raise ValueError("Weights must have the same shape as the center point.")
            weights = weights / weights.mean()
            weights = weights / torch.prod(weights.pow(1.0 / len(weights)))

        rescaling_factor = (1 + buffer) * 2.0
        tr_lb = torch.clamp(x_center - weights * self.length * rescaling_factor, 0.0, 1.0)
        tr_ub = torch.clamp(x_center + weights * self.length * rescaling_factor, 0.0, 1.0)

        return tr_lb, tr_ub, x_center

    def get_trust_region_data(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        X_pending: torch.Tensor | None = None,
        buffer: float = 0.0,
        weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Get the trust region data for the TuRBO algorithm."""
        # Get best point and create trust region
        tr_lb, tr_ub, _ = self.get_trust_region_bounds(X, Y, buffer=buffer, weights=weights)  # pyright: ignore[reportArgumentType]

        # Filter points within trust region + buffer for training
        mask = torch.all((tr_lb <= X) & (tr_ub >= X), dim=1)
        X_train = X[mask]
        Y_train = Y[mask]

        if X_pending is not None:
            X_pending = X_pending[mask]

        return X_train, Y_train, X_pending

    def generate_batch(
        self,
        model: SingleTaskGP,  # GP model
        X: torch.Tensor,  # Evaluated points on the domain [0, 1]^d
        Y: torch.Tensor,  # Function values
        batch_size: int,
        X_pending: torch.Tensor | None = None,
        acqf: Literal["ei", "ts"] = "ts",
        acqf_kwargs: dict[str, Any] | None = None,
        y_is_oriented: bool = False, # flag to explicitly mark if incoming Y is already transformed for “maximization”
    ) -> torch.Tensor:
        """Generate a batch of candidates within the current trust region."""
        if acqf_kwargs is None:
            acqf_kwargs = {}

        assert acqf in ("ts", "ei")
        assert X.min() >= 0.0
        assert X.max() <= 1.0
        assert torch.all(torch.isfinite(Y))

        dtype = X.dtype
        device = X.device

        Y = Y.to(dtype=dtype, device=device)
        model = model.to(dtype=dtype, device=device)

        if acqf == "ts" and "n_candidates" not in acqf_kwargs:
            acqf_kwargs["n_candidates"] = min(5000, max(2000, 200 * X.shape[-1]))

        # Scale the TR to be proportional to the lengthscales
        weights = model.covar_module.base_kernel.lengthscale.squeeze().detach()  # pyright: ignore[reportCallIssue, reportAttributeAccessIssue]
        tr_lb, tr_ub, x_center = self.get_trust_region_bounds(
            X,
            Y,
            weights=weights,
            maximize=True if y_is_oriented else None, # if y_is_oriented is True, then we are maximizing as Y is already oriented for maximization
        )

        if acqf == "ts":
            n_candidates = acqf_kwargs.get("n_candidates", min(5000, max(2000, 200 * X.shape[-1])))
            dim = X.shape[-1]
            sobol = SobolEngine(dim, scramble=True)
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

            # Sample on the candidate points
            thompson_sampling = MaxPosteriorSampling(model=model, replacement=False)
            with torch.no_grad():  # We don't need gradients when using TS
                X_next = thompson_sampling(X_cand, num_samples=batch_size)

        elif acqf == "ei":
            best_f = Y.max() if y_is_oriented else (Y.max() if self.maximize else -Y.min()) # else is just fallback, should not be needed as the Y passed in will have the correct orientation
            ei = qLogExpectedImprovement(model, best_f, X_pending=X_pending)
            X_next, _ = optimize_acqf(
                ei,
                bounds=torch.stack([tr_lb, tr_ub]),
                q=batch_size,
                **acqf_kwargs,
            )

        return X_next


class TurboGenerationNode(ExternalGenerationNode):
    """A generation node that uses the TuRBO algorithm to generate a set of
    candidate designs.
    """

    def __init__(
        self,
        model_options: dict[str, Any],
        batch_size: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        acqf: Literal["ts", "ei"] = "ts",
        acqf_kwargs: dict[str, Any] | None = None,
        name: str = "TurboGenerationNode",
        maximize: bool = True,
    ) -> None:
        """Initialize the generation node.

        Args:
            model_options: Options to pass to the GP model.
            batch_size: The batch size for generating new candidates.
            device: The device to use for the generation node.
            dtype: The dtype to use for the generation node.
            acqf: Acquisition function to use ("ts" for Thompson sampling,
                "ei" for Expected Improvement).
            acqf_kwargs: Keyword arguments for the acquisition function.
            name: The name of the generation node.
            maximize: Whether the generation node is maximizing or minimizing the objective
        """
        if acqf_kwargs is None:
            acqf_kwargs = {}

        super().__init__(name=name)

        self.device = device if device is not None else torch.device("cpu")
        self.dtype = dtype if dtype is not None else torch.double

        self.model_options = model_options
        self.batch_size = batch_size
        self.acqf = acqf
        self.acqf_kwargs = acqf_kwargs
        self.state: TurboState | None = None
        self.X_turbo: torch.Tensor | None = None
        self.Y_turbo: torch.Tensor | None = None
        self.parameters: list[RangeParameter] | None = None
        self.maximize = maximize

        self.sobol = None

    def update_generator_state(self, experiment: Experiment, data: Data) -> None:
        """Update the state of the generator with the experiment and data.

        Args:
            experiment: The experiment object.
            data: The data object.
        """
        search_space = experiment.search_space

        if any(not isinstance(p, RangeParameter) for p in search_space.parameters.values()):
            raise NotImplementedError("This method only supports RangeParameters in the search space.")

        parameter_names = list(search_space.parameters.keys())
        metric_names = list(experiment.optimization_config.metrics.keys())  # pyright: ignore[reportOptionalMemberAccess]

        if self.parameters is None:
            self.parameters = list(search_space.parameters.values())  # pyright: ignore[reportAttributeAccessIssue]

        if self.sobol is None:
            self.sobol = SobolEngine(len(parameter_names), scramble=True)

        # Initialize TuRBO state and data if it's the first call
        if self.state is None:
            self.state = TurboState(dim=len(parameter_names), batch_size=self.batch_size, maximize=self.maximize)

        if len(metric_names) != 1:
            raise ValueError("This generation node only supports a single metric.")

        # Get the data for the completed trials.
        num_completed_trials = len(experiment.trials_by_status[TrialStatus.COMPLETED])
        X = torch.zeros([num_completed_trials, len(parameter_names)], dtype=self.dtype, device=self.device)
        Y = torch.zeros([num_completed_trials, 1], dtype=self.dtype, device=self.device)

        for t_idx, trial in experiment.trials.items():
            if trial.status == TrialStatus.COMPLETED:
                trial_parameters = trial.arm.parameters  # pyright: ignore[reportAttributeAccessIssue]
                X[t_idx, :] = torch.tensor([trial_parameters.get(p, 0) for p in parameter_names], dtype=self.dtype)
                trial_df = data.df[data.df["trial_index"] == t_idx]
                filtered_df = trial_df[trial_df["metric_name"] == metric_names[0]]
                if not filtered_df.empty:
                    Y[t_idx, 0] = torch.tensor(filtered_df["mean"].item(), dtype=self.dtype)
                else:
                    Y[t_idx, 0] = torch.tensor(float("nan"), dtype=self.dtype)  # Handle missing data

        # Normalize X to [0, 1]^d
        X_normalized = self.to_unit_cube(X)
        Y_new = Y[~torch.isnan(Y).any(dim=1)]  # Filter out NaN values
        X_normalized = X_normalized[~torch.isnan(X_normalized).any(dim=1)]
        self.state = self.state.update_state(Y_next=Y_new)

        self.X_turbo = X_normalized
        self.Y_turbo = Y_new

    def to_unit_cube(self, X: torch.Tensor) -> torch.Tensor:
        """Convert a tensor of parameters to the unit cube."""
        if self.parameters is None:
            raise RuntimeError("Generator state not initialized. Call update_generator_state first.")

        lower_bounds = torch.tensor(
            [p.lower for p in self.parameters],
            dtype=self.dtype,
            device=self.device,
        )

        upper_bounds = torch.tensor(
            [p.upper for p in self.parameters],
            dtype=self.dtype,
            device=self.device,
        )
        return (X - lower_bounds) / (upper_bounds - lower_bounds)

    def from_unit_cube(self, X: torch.Tensor) -> torch.Tensor:
        """Convert a tensor of parameters from the unit cube to the original space."""
        if self.parameters is None:
            raise RuntimeError("Generator state not initialized. Call update_generator_state first.")

        lower_bounds = torch.tensor(
            [p.lower for p in self.parameters],
            dtype=self.dtype,
            device=self.device,
        )
        upper_bounds = torch.tensor(
            [p.upper for p in self.parameters],
            dtype=self.dtype,
            device=self.device,
        )
        return X * (upper_bounds - lower_bounds) + lower_bounds

    @override
    def get_next_candidate(self, pending_parameters: list[TParameterization]) -> TParameterization:
        """Get the parameters for the next candidate configuration to evaluate.

        Args:
            pending_parameters: A list of parameters of the candidates pending
                evaluation.

        Returns:
            A dictionary mapping parameter names to parameter values for the next
            candidate suggested by the method.
        """
        if self.X_turbo is None or self.Y_turbo is None or self.state is None or self.parameters is None:
            raise RuntimeError("Generator state not initialized. Call update_generator_state first.")

        if self.state.restart_triggered:
            raise TurboConvergedError(
                "TuRBO has converged (trust region below minimum). "
                "To continue optimization, create a new TurboGenerationNode instance with fresh initial points."
            )

        if len(pending_parameters) > 0:
            X_pending = torch.zeros(len(pending_parameters), len(self.parameters), dtype=self.dtype, device=self.device)
            parameter_names = [p.name for p in self.parameters]
            for i, pending in enumerate(pending_parameters):
                X_pending[i, :] = torch.tensor(
                    [pending.get(name, 0.0) for name in parameter_names],
                    dtype=self.dtype,
                    device=self.device,
                )

            X_pending = self.to_unit_cube(X_pending)
        else:
            X_pending = None

        X_train, Y_train = self.X_turbo, self.Y_turbo
        # Orient objective once so acquisition always solves a maximization problem.
        # flips the signs for minimization problems, and sets y_is_oriented=True to indicate that Y passed to generate_batch is already oriented for maximization
        Y_for_model = Y_train if self.maximize else -Y_train # Acqf samples from a model aligned with objective direction


        # Fit a GP model with Standardize transform (handles standardization automatically)
        # The posterior is in original scale, so we pass original Y to generate_batch
        model = fit_gp(
            X_train,
            Y_for_model,
            normalize_inputs=False,  # Already normalized to [0,1]
            standardize_outputs=True,
            max_cholesky_size=self.model_options.get("max_cholesky_size", float("inf")),
        )

        # Create a batch using the same oriented target used for GP fitting.
        X_next = self.state.generate_batch(
            model=model,
            X=X_train,
            X_pending=X_pending,
            Y=Y_for_model,
            batch_size=self.batch_size,
            acqf=self.acqf,  # pyright: ignore[reportArgumentType]
            acqf_kwargs=self.acqf_kwargs,
            y_is_oriented=True, # flag to indicate that Y passed to generate_batch is already oriented(flipped if minimize) for maximization from the Y_for_model variable
        )

        X_next_unnormalized = self.from_unit_cube(X_next)

        # Convert the sample to a parameterization.
        return dict(
            zip(
                [p.name for p in self.parameters],
                X_next_unnormalized.ravel().detach().tolist(),
                strict=True,
            )
        )
