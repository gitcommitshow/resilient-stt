"""Pytest hooks: disable third-party telemetry before any tests run."""

from resilient_stt.core.privacy import disable_dependency_telemetry

disable_dependency_telemetry()
