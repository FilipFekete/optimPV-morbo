Change Log
==========
All notable changes to this project will be documented in this file.

v1.03 - 2025-08-14 - VMLC-PV
-----------------------------
- Upgrading the ax-platform version to 1.0.0.
- axBOtorchOptimizer: Removing the optimize(batch=True) that used the runner in older versions of optimPV. Mostly because it did not provide any significant benefits and added unnecessary complexity and that the new ax-platform broke the compatibility with the old code. Multiple small fixes to the code to make it compatible with the new ax-platform version 1.0.0. Removed the parallel_agent option and replaced it with a parallel option that will run everything in parallel.
- PymooOptimizer: Initial implementation of the PymooOptimizer class for single and multi-objective optimization using the [pymoo](https://github.com/anyoptimization/pymoo) library. This class provide new optimization capabilities withe the evolutionary/genetic algorithms and other algorithms available in the Pymoo library. The class is designed to have a similar interface to the AxOptimizer class, allowing for easy integration with existing code. The class supports both single and multi-objective optimization, and can be used with the same parameters and agents as the AxOptimizer class. The class also supports parallel optimization using the Pymoo library's parallelization capabilities. However, for not it only support float as value_type for the parameters. 
- posterior: added the 'pymoo' optimizer as an options for the **plot_density_exploration** function. This allows to plot the density exploration of the optimization process using the Pymoo library's optimization results.
- New Notebooks and created tests for the new functionalities.
- logger: Added logger inspired but the one in the Ax library. 
- SuggestOnlyAgent: Added the SuggestOnlyAgent class to provide a simple way to suggest new parameters without running the simulation/experiment and using a known dataset. This is useful when doing design of experiments (DoE) in multiple steps. It can be used with the axBOtorchOptimizer and the PymooOptimizer classes to suggest new parameters based on the previous results. 
- EGBO: Updated the EGBOAcquisition class to work with the new ax-platform version 1.0.0. 
- RateEqAgent: small fixes for the case Gfracs = None we where not taking the right time axis to take into account the pump frequency, this was done properly when we had several Gfracs. 
- RateEqModel: Added two new model based on the work by [Kober-Czerny et al. 2025](https://doi.org/10.1103/PRXEnergy.4.013001) and [M. Simmonds](https://github.com/MaximSimmonds-HZB/MAPI-FAPI-fitting).
- Pumps: fixing the initial_carrier_density to convert the N0 into proper generation rate.

v1.02 - 2025-05-20 - VMLC-PV
-----------------------------
- Fixing the ax-platform version to 0.5.0 as version 1.0.0 breaks the compatibility with the current code. This is a temporary fix until the code is updated to be compatible with ax-platform 1.0.0.
- scipyOptimizer: Adding the scipyOptimizer class to provide a simple interface for optimization using the scipy library. The class is typically for single-objective optimization problems and supports various optimization algorithms available in the [scipy](https://github.com/scipy/scipy) library.
- EmceeOptimizer: Adding the EmceeOptimizer class to provide an interface for optimization using the [emcee](https://github.com/dfm/emcee) library. The class is designed to work with the Bayesian inference methods provided by emcee, allowing for efficient sampling of the parameter space.

v1.01 - 2025-04-29 - VMLC-PV
-----------------------------
- BaseAgent.py: Modified the params_w so such that param.type = 'fixed' are not transformed.
- JVAgent.py, IMPSAgent,py, ImpedanceAgent.py, HysteresisAgent.py, CVAgent.py: Added a safety check to ensure there is no infinite loop in the __init__ method when checking for the input simulation files for SIMsalabim.
- Small docstrings fixes.


v1 - 2025-04-10 - VMLC-PV
---------------------------
- Initial release of the project.