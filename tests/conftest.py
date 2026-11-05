from __future__ import annotations

import pytest

from fraudplatform.config import SimulationConfig, load_simulation_config
from fraudplatform.simulation.generator import SimulationResult, simulate


@pytest.fixture(scope="session")
def sim_config() -> SimulationConfig:
    return load_simulation_config()


@pytest.fixture(scope="session")
def tiny(sim_config) -> SimulationResult:
    return simulate(sim_config, "tiny")


@pytest.fixture(scope="session")
def small(sim_config) -> SimulationResult:
    """~50k transactions: large enough for rates and scenario coverage, still a few seconds."""
    return simulate(sim_config, "small")
