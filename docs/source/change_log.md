Change Log
==========
All notable changes to this project will be documented in this file.

v1.01 - 2025-04-29 - VMLC-PV
-----------------------------
- BaseAgent.py: Modified the params_w so such that param.type = 'fixed' are not transformed.
- JVAgent.py, IMPSAgent,py, ImpedanceAgent.py, HysteresisAgent.py, CVAgent.py: Added a safety check to ensure there is no infinite loop in the __init__ method when checking for the input simulation files for SIMsalabim.
- Small docstrings fixes.


v1 - 2025-04-10 - VMLC-PV
---------------------------
- Initial release of the project.