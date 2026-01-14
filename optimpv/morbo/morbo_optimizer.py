#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
MorboOptimizer: Adapter that lets optimPV agents run with the MORBO/TRBO loop.

The interface mirrors other optimizers in the repo: pass FitParam objects and
Agent instances and call optimize(). Metrics are aggregated across
agents using the BaseAgent helpers so naming/minimize flags match Ax/Pymoo usage.
"""
from __future__ import annotations

import copy
import time
from typing import Dict, Iterable, List, Optional, Tuple
import numpy as np 

import torch
from botorch.utils.sampling import draw_sobol_samples

from optimpv.general.BaseAgent import BaseAgent
from optimpv.morbo.gen import TS_select_batch_MORBO
from optimpv.morbo.state import TRBOState
from optimpv.morbo.trust_region import TurboHParams


class MorboOptimizer(BaseAgent):
    """
    Adapter that lets optimPV agents run inside the MORBO/TRBO loop.

    The interface mirrors other optimizers in the repo so callers can swap
    implementations without changing how agents are constructed or invoked.
    FitParams define the search space, agents provide metrics, and the optimizer
    handles Sobol seeding, trust-region management, and hypervolume tracking.

    Parameters
    ----------
    params : list of FitParam() objects
        List of FitParam objects (fixed params allowed; they are held constant), by default None.
    agents : list of Agent() objects
        Agents exposing run_Ax and metric metadata (all_agent_metrics/minimize), by default None.
    max_evals : int
        Total evaluation budget (including initial Sobol points).
    batch_size : int
        Batch size for MORBO candidate generation.
    n_initial_points : int
        Number of Sobol points to seed each run.
    reference_point : list[float] | None
        Reference point in the (possibly sign-flipped) objective space for HV.
        Required for multi-objective HV computations; optional for single obj.
    tr_hparam_overrides : dict
        Values to override defaults in TurboHParams. Supported keys include:
        - TR sizing/tolerances: length_init, length_min, length_max,
          success_streak, failure_streak (numeric; default max(dim//3, 10)),
          min_tr_size, max_tr_size, n_trust_regions
        - Sampling: raw_samples, qmc, sample_subset_d, trunc_normal_perturb
        - Hypervolume: hypervolume (bool), max_reference_point (sign-adjusted
          reference_point), use_approximate_hv_computations,
          approximate_hv_alpha, restart_hv_scalarizations
        - Misc: track_history, trim_trace, use_ard, max_cholesky_size,
          tabu_tenure, decay_restart_length_alpha, switch_strategy_freq,
          use_noisy_trbo, use_simple_rff
          
    torch_dtype : torch.dtype
    torch_device : torch.device | None
    seed : int | None
        If provided, sets both torch and numpy seeds for reproducibility.
    verbose : bool
        If True, emits per-iteration timing and HV logs from TRBOState.
    """

    def __init__(
        self,
        params=None,
        agents=None,
        max_evals: int = 50,
        batch_size: int = 1,
        n_initial_points: int = 10,
        reference_point: Optional[List[float]] = None,
        tr_hparam_overrides: Optional[Dict] = None,
        torch_dtype: torch.dtype = torch.double,
        torch_device: Optional[torch.device] = None,
        seed: Optional[int] = None,
        verbose: bool = False,
    ):
        super().__init__()
        if params is None or agents is None:
            raise ValueError("params and agents must be provided.")
        if not isinstance(agents, list):
            agents = [agents]

        self.params = params
        self.agents = agents
        for agent in self.agents:
            agent.params = self.params

        self.max_evals = int(max_evals)
        self.batch_size = int(batch_size)
        self.n_initial_points = int(n_initial_points)
        self.reference_point = reference_point
        self.tr_hparam_overrides = tr_hparam_overrides or {}
        self.torch_dtype = torch_dtype
        # Default to CUDA when available; MORBO models and sampling benefit from GPU.
        self.device = (
            torch_device
            if torch_device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.verbose = verbose

        if seed is not None:
            # Keep both torch and numpy reproducible when a seed is provided.
            torch.manual_seed(seed)
            try:
                np.random.seed(seed)

            except Exception:
                pass

        self.all_metrics, self.all_minimize = self.create_metrics_list()
        self._setup_bounds()

        self.history = None  # filled after optimize()

    # ------------------------------------------------------------------ setup
    def _setup_bounds(self) -> None:
        """
        Create the descaled bounds tensor for MORBO's internal search space.

        MORBO/TRBO operate on descaled parameters, so we strip out fixed params,
        transform FitParam bounds, and prep the hypervolume reference point with
        the same sign convention used on metrics.
        """
        bounds_raw = [[], []]
        self.opt_param_names: List[str] = []
        for p in self.params:
            if p.type != "fixed":
                bounds_raw[0].append(p.bounds[0])
                bounds_raw[1].append(p.bounds[1])
                self.opt_param_names.append(p.name)

        if len(self.opt_param_names) == 0:
            raise ValueError("No free parameters to optimize.")

        # uses BaseAgent bounds_descale to match FitParam() oject descaling
        descale_bounds = self.bounds_descale(bounds_raw, self.params)
        # converts to tensor 
        self.bounds_tensor = torch.tensor(
            descale_bounds, device=self.device, dtype=self.torch_dtype
        )
        # set the number of dimensions to the length of the parameters
        self.dim = len(self.opt_param_names)
        # get the number of objectives 
        self.num_objectives = len(self.all_metrics)
        self.num_outputs = self.num_objectives  # no constraints yet

        if self.num_objectives > 1 and self.reference_point is None:
            raise ValueError(
                "reference_point is required for multi-objective MORBO runs."
            )
        
        if self.reference_point is not None:
            if len(self.reference_point) != self.num_objectives:
                raise ValueError(
                    "reference_point length must match number of objectives."
                )
            # Apply the same sign flip we use on objectives for minimize metrics.
            pair_count = min(len(self.reference_point), len(self.all_minimize))
            ref_point_signed = []
            for idx in range(pair_count):
                rp = float(self.reference_point[idx])
                minimize = self.all_minimize[idx]
                sign = -1.0 if minimize else 1.0
                ref_point_signed.append(sign * rp)
            self.ref_point_signed = ref_point_signed
            self.ref_point_tensor = torch.tensor(
                self.ref_point_signed, device=self.device, dtype=self.torch_dtype
            )

        else:
            self.ref_point_tensor = None
            self.ref_point_signed = None

    # ---------------------------------------------------------------- evaluate
    def _vector_to_param_dict(self, x: Iterable[float]) -> Dict:
        """Map a flat vector to a parameter dict (descaled space)."""
        param_dict = {}
        idx = 0
        for p in self.params:
            if p.type == "fixed":
                param_dict[p.name] = p.value
            else:
                param_dict[p.name] = float(x[idx])
                idx += 1
        return param_dict

    def _evaluate_agents(self, X: torch.Tensor) -> torch.Tensor:
        """
        Evaluate agents on a batch of points and align metric signs for MORBO.

        Each row of X is converted back to parameter dictionaries (including
        fixed params), run through every agent, and aggregated in metric order.
        Metrics flagged for minimization are sign-flipped so MORBO can treat all
        objectives as maximize internally.
        """
        results = []
        for row in X:
            p_dict = self._vector_to_param_dict(row.tolist())
            agg: Dict[str, float] = {}
            for agent in self.agents:
                agg.update(agent.run_Ax(copy.deepcopy(p_dict)))
            y = []
            pair_count = min(len(self.all_metrics), len(self.all_minimize))
            for idx in range(pair_count):
                name = self.all_metrics[idx]
                minimize = self.all_minimize[idx]
                val = agg[name]
                # MORBO assumes larger is better; flip sign for minimize metrics.
                y.append(-float(val) if minimize else float(val))
            results.append(y)
        return torch.tensor(results, device=self.device, dtype=self.torch_dtype)

    def _unflip_metrics(self, Y: torch.Tensor) -> torch.Tensor:
        """Convert sign-flipped metrics back to their original orientation."""
        signs = torch.tensor(
            [-1.0 if minimize else 1.0 for minimize in self.all_minimize],
            device=Y.device,
            dtype=Y.dtype,
        )
        return Y * signs
    
    # TODO Helper method to return raw metric history as pd.DataFrame
    # TODO Helper method to return parameter history in physical units as a DataFrame
    # TODO Helper method Ax-style for rank-based balance across objectives

    # ------------------------------------------------------------- main driver
    def optimize(self) -> Dict:
        """
        Run MORBO/TRBO end-to-end and return histories plus Pareto fronts.

        Mirrors MORBO optimizer, run_one_replication.py: Sobol seeding, iterative candidate generation
        and agent evaluation, trust-region updates/restarts, and hypervolume
        tracking. Returnes history, keeps both flipped and raw metrics so
        downstream plotting code can reuse it without re-running agents.
        """
        tr_kwargs = copy.deepcopy(self.tr_hparam_overrides)

        # Prevent duplicate kwargs when we set these explicitly.
        for k in ("batch_size", "n_initial_points", "max_reference_point", "verbose"):
            tr_kwargs.pop(k, None)

        # Decide whether to use hypervolume which is default for multi-objective
        hypervolume_flag = tr_kwargs.pop("hypervolume", self.num_objectives > 1)

        # Avoid None failure_streak (ensure its numeric) which breaks comparison in TR updates.
        # If not provided, set to max(dim//3, 10) -> (either one third of the dimensions or 10, whichever is larger) as in original TRBO. 
        tr_kwargs.setdefault("failure_streak", max(self.dim // 3, 10))

        # Build per-TR hyperparameters plus the overrides and the shared TRBO state container.
        # When you look at the TurboHParams in morbo/trust_region.py, you'll see all the defaults.
        # To keep it flexible and simple, everything not explicitly set here is pulled from tr_hparam_overrides, that can be passed upon initialization.
        tr_hparams = TurboHParams(
            batch_size=self.batch_size,
            n_initial_points=self.n_initial_points,
            max_reference_point=self.ref_point_signed,
            verbose=self.verbose,
            hypervolume=hypervolume_flag,
            **tr_kwargs, 
        )
        # TRBOstate holds e.g. dimensions, eval budget, outputs, bounds tensor, TR hyperparams
        trbo_state = TRBOState(
            dim=self.dim,
            max_evals=self.max_evals,
            num_outputs=self.num_outputs,
            num_objectives=self.num_objectives,
            bounds=self.bounds_tensor,
            tr_hparams=tr_hparams,
            constraints=None,
            objective=None,
        )

        # Initial Sobol design:
        # - Sample Sobol points within bounds (respecting max_evals cap)
        # - Evaluate agents to seed Y
        # - Log restart points so all TRs start with identical training data
        n_points = min(self.n_initial_points, self.max_evals)
        X_init = draw_sobol_samples(
            bounds=self.bounds_tensor, n=n_points, q=1
        ).squeeze(1)
        # evaluate agents on those points (with sign flips for minimize)
        Y_init = self._evaluate_agents(X_init)
        # add these points to the state/history, marking their TR index as 0
        trbo_state.update(
            X=X_init,
            Y=Y_init,
            new_ind=torch.full(
                (X_init.shape[0],), 0, dtype=torch.long, device=self.device
            ),
        )
        # record the points for potential restarts
        trbo_state.log_restart_points(X=X_init, Y=Y_init)

        # Seed each trust region with the same initial design so every TR can
        # fit a model before any Thompson sampling decisions are made.
        for tr_idx in range(tr_hparams.n_trust_regions):
            trbo_state.initialize_standard(
                tr_idx=tr_idx,
                restart=False,
                switch_strategy=False,
                X_init=X_init,
                Y_init=Y_init,
            )

        # Share the seed data across TRs
        trbo_state.update_data_across_trs()

        # Mark Sobol seeds with -2 so later restarts/switches are distinguishable.
        trbo_state.TR_index_history.fill_(-2)

        # Logs
        n_evals, pareto_X, pareto_Y, hv_trace = [], [], [], []
        all_tr_indices: List[int] = []
        fit_times, gen_times = [], []  # runtime diagnostics per iteration


        # Main MORBO/TRBO loop until max_evals is reached
        start_time = time.time()
        while trbo_state.n_evals < self.max_evals:
            # 1) Propose batch via Thompson sampling over active trust regions.
            start_gen = time.time()
            selection = TS_select_batch_MORBO(trbo_state=trbo_state)
            gen_times.append(time.time() - start_gen)
            if trbo_state.tr_hparams.verbose:
                print(f"Time spent on generating candidates: {gen_times[-1]:.1f} seconds")

            X_cand = selection.X_cand
            tr_indices = selection.tr_indices
            all_tr_indices.extend(tr_indices.tolist())
            trbo_state.tabu_set.log_iteration()
            Y_cand = self._evaluate_agents(X_cand)

            # 2) Fit/update TRs with new observations and decide on restarts.
            start_fit = time.time()
            trbo_state.update(X=X_cand, Y=Y_cand, new_ind=tr_indices)
            should_restart = trbo_state.update_trust_regions_and_log(
                X_cand=X_cand,
                Y_cand=Y_cand,
                tr_indices=tr_indices,
                batch_size=self.batch_size,
                verbose=self.verbose,
            )
            fit_times.append(time.time() - start_fit)

            # Handle strategy switching (HV vs scalarization) if configured, apply restarts
            switch_strategy = trbo_state.check_switch_strategy()
            if switch_strategy:
                should_restart = [True for _ in should_restart]
            if any(should_restart):
                for i in range(trbo_state.tr_hparams.n_trust_regions):
                    if not should_restart[i]:
                        continue
                    # Limit restart seed size to remaining budget.
                    n_points_restart = min(
                        trbo_state.tr_hparams.n_restart_points,
                        self.max_evals - trbo_state.n_evals,
                    )
                    if n_points_restart <= 0:
                        break
                    trbo_state.TR_index_history[
                        trbo_state.TR_index_history == i
                    ] = -1
                    init_kwargs = {}
                    if trbo_state.tr_hparams.restart_hv_scalarizations:
                        # Generate a new center via scalarized HV sampling and
                        # immediately fold it into the shared dataset.
                        X_center = trbo_state.gen_new_restart_design()
                        Y_center = self._evaluate_agents(X_center)
                        init_kwargs["X_init"] = X_center
                        init_kwargs["Y_init"] = Y_center
                        init_kwargs["X_center"] = X_center
                        trbo_state.update(
                            X=X_center,
                            Y=Y_center,
                            new_ind=torch.tensor(
                                [i], dtype=torch.long, device=self.device
                            ),
                        )
                        trbo_state.log_restart_points(
                            X=X_center, Y=Y_center
                        )
                    trbo_state.initialize_standard(
                        tr_idx=i,
                        restart=True,
                        switch_strategy=switch_strategy,
                        **init_kwargs,
                    )
                    if trbo_state.tr_hparams.restart_hv_scalarizations:
                        trbo_state.update_data_across_trs()

            # HV/Pareto logging per interation 
            n_evals.append(trbo_state.n_evals.item())
            # Reuses TRBOState’s HV, doesn’t recompute/denoise
            if trbo_state.hv is not None:
                hv_val = (
                    trbo_state.hv.item()
                    if hasattr(trbo_state.hv, "item")
                    else float(trbo_state.hv)
                )
                pareto_X.append(trbo_state.pareto_X.tolist())
                pareto_Y.append(trbo_state.pareto_Y.tolist())
                hv_trace.append(hv_val)
            else:
                pareto_X.append([])
                pareto_Y.append([])
                hv_trace.append(0.0)

            trbo_state.update_data_across_trs()

        end_time = time.time()
        if self.verbose:
            print(f"Total time: {end_time - start_time:.1f} seconds")

        # Detailed traces (with both flipped and raw metrics) so callers
        # can plot HV, pareto fronts, and timing without recomputing anything.
        self.history = {
            "n_evals": n_evals, # list of evaluation count (int)
            "X_history": trbo_state.X_history.cpu(), # torch.Tensor (n_evals, dim) in descaled/internal space
            "metric_history": trbo_state.Y_history.cpu(), # torch.Tensor (n_evals, num_objectives) in MORBO internal space (minimize metrics are multiplied by -1)
            "metric_history_raw": self._unflip_metrics(trbo_state.Y_history).cpu(), # torch.Tensor, unflipped to original values (losses are poistive)
            "pareto_X": pareto_X, # list of lists (per iteration) in internal space
            "pareto_Y": pareto_Y, # list of lists (per iteration) sign-flipped
            "pareto_Y_raw": [ # list of lists (per iteration), unflipped
                self._unflip_metrics(torch.tensor(y, dtype=self.torch_dtype)).tolist()
                if y
                else []
                for y in pareto_Y
            ],
            "hv": hv_trace, # list of floats (hypervolume trace)
            "tr_indices": all_tr_indices, 
            "fit_times": fit_times,
            "gen_times": gen_times,
        }
        return self.history # returns dictionary with arrays/lists

        # To see true losses, use metric_history_raw or pareto_Y_raw
        # To see best params, use X_history + _vector_to_param_dict + params_rescale

__all__ = ["MorboOptimizer"]
