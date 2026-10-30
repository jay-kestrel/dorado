"""Runnable scenario cases, discovered by the ``scenarios`` registry.

Each module defines NAME, DESCRIPTION, and ``run(quick, seed) -> dict``
(one metrics record); :mod:`scenarios.core` provides the shared timing,
memory, analysis, and record-assembly contract.
"""
