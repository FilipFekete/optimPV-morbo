import importlib.util
import sys
import numpy as np
import time

import ray  # Import ray normally
from ray import tune
# Load your custom ax_search_custom.py
custom_module_name = "ax_search_custom"
custom_file = "ax_search_custom.py"

spec = importlib.util.spec_from_file_location(custom_module_name, custom_file)
ax_search_custom = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ax_search_custom)

# Now import AxSearch from the custom module instead of ray's version
AxSearch = ax_search_custom.AxSearch

# Use AxSearch while still using ray
def landscape(x):
    """
    Hartmann 6D function containing 6 local minima.
    It is a classic benchmark for developing global optimization algorithms.
    """
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array(
        [
            [10, 3, 17, 3.5, 1.7, 8],
            [0.05, 10, 17, 0.1, 8, 14],
            [3, 3.5, 1.7, 10, 17, 8],
            [17, 8, 0.05, 10, 0.1, 14],
        ]
    )
    P = 10 ** (-4) * np.array(
        [
            [1312, 1696, 5569, 124, 8283, 5886],
            [2329, 4135, 8307, 3736, 1004, 9991],
            [2348, 1451, 3522, 2883, 3047, 6650],
            [4047, 8828, 8732, 5743, 1091, 381],
        ]
    )
    y = 0.0
    for j, alpha_j in enumerate(alpha):
        t = 0
        for k in range(6):
            t += A[j, k] * ((x[k] - P[j, k]) ** 2)
        y -= alpha_j * np.exp(-t)
    return y


def objective(config):
    for i in range(config["iterations"]):
        x = np.array([config.get("x{}".format(i + 1)) for i in range(6)])
        tune.report(
            {"timesteps_total": i, "landscape": landscape(x), "l2norm": np.sqrt((x ** 2).sum())}
        )
        time.sleep(0.02)


search_space = {
    "iterations":100,
    "x1": tune.uniform(0.0, 1.0),
    "x2": tune.uniform(0.0, 1.0),
    "x3": tune.uniform(0.0, 1.0),
    "x4": tune.uniform(0.0, 1.0),
    "x5": tune.uniform(0.0, 1.0),
    "x6": tune.uniform(0.0, 1.0)
}

algo = AxSearch(
    parameter_constraints=["x1 + x2 <= 2.0"],
    outcome_constraints=["l2norm <= 1.25"],
)


algo = tune.search.ConcurrencyLimiter(algo, max_concurrent=4)

num_samples = 100
stop_timesteps = 200

tuner = tune.Tuner(
    objective,
    tune_config=tune.TuneConfig(
        metric="landscape",
        mode="min",
        search_alg=algo,
        num_samples=num_samples,
    ),
    run_config=tune.RunConfig(
        name="ax",
        stop={"timesteps_total": stop_timesteps}
    ),
    param_space=search_space,
)
results = tuner.fit()

