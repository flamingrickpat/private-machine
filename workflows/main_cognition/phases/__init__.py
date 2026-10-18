"""The phases of the main cognition cycle, one module for each phase.

``workflows/main_cognition/steps.py`` is the Python role of every phase. It
reads the shared :class:`~workflows.main_cognition.cycle_work.CycleWork`,
calls the ``run`` function of the phase module, and writes the result back.
"""
