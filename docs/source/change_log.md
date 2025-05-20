Change Log
==========
All notable changes to this project will be documented in this file.
v1.02 - 2025-05-20 - VMLC-PV
-----------------------------
- Fixing the ax-platform version to 0.5.0 as the version 1.0.0 break the compatibility with the current code. This is a temporary fix until the code is updated to be compatible with ax-platform 1.0.0.

v1.01 - 2025-04-29 - VMLC-PV
-----------------------------
- BaseAgent.py: Modified the params_w so such that param.type = 'fixed' are not transformed.
- JVAgent.py, IMPSAgent,py, ImpedanceAgent.py, HysteresisAgent.py, CVAgent.py: Added a safety check to ensure there is no infinite loop in the __init__ method when checking for the input simulation files for SIMsalabim.
- Small docstrings fixes.


v1 - 2025-04-10 - VMLC-PV
---------------------------
- Initial release of the project.