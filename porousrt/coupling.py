"""Backend-neutral sequential reactive-transport coupling.

This module owns the reusable SNIA orchestration shared by the cases.  A case
provides three small modeling interfaces: conservative transport, local
chemistry, and quasi-steady variable-density flow.  The coupler handles their
ordering, reaction/solution-volume mass bookkeeping, and the slower flow
refresh cadence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

import numpy as np
from numpy.typing import NDArray

from .transport import ComponentBoundaryConditions


FloatArray = NDArray[np.float64]


class ReactiveState(Protocol):
    """Minimum state supplied by a local geochemical model."""

    concentrations: FloatArray
    porosity: FloatArray
    density_kg_m3: FloatArray
    viscosity_pa_s: FloatArray

    @property
    def solid_mass_release_rate_kg_m3_s(self) -> FloatArray:
        """Net mineral-to-fluid mass rate, positive for net dissolution."""


StateT = TypeVar("StateT", bound=ReactiveState)


class ChemistryModel(Protocol[StateT]):
    """Local chemistry interface required by :class:`SequentialCoupler`."""

    initial_state: StateT
    inlet_concentrations: FloatArray | ComponentBoundaryConditions

    def react(
        self,
        transported_concentrations: FloatArray,
        porosity: FloatArray,
        dt: float,
    ) -> StateT: ...


class TransportModel(Protocol):
    """Conservative component-transport interface."""

    def advance(
        self,
        concentrations: FloatArray,
        porosity: FloatArray,
        u: FloatArray,
        v: FloatArray,
        *,
        dt: float,
        molecular_diffusivity_m2_s: float,
        inlet_concentrations: FloatArray | ComponentBoundaryConditions,
    ) -> FloatArray: ...


class FlowModel(Protocol):
    """Quasi-steady flow interface."""

    def solve(
        self,
        porosity: FloatArray,
        density_kg_m3: FloatArray,
        viscosity_pa_s: FloatArray,
        mass_balance_source_kg_m3_s: FloatArray,
    ) -> tuple[FloatArray, FloatArray, FloatArray, dict[str, float]]: ...


@dataclass
class CoupledState(Generic[StateT]):
    """One synchronized chemistry, flow, and coupling state."""

    time: float
    chemistry: StateT
    u: FloatArray
    v: FloatArray
    pressure_pa: FloatArray
    mass_source_kg_m3_s: FloatArray
    maximum_porosity_change: float
    flow_diagnostics: dict[str, float]


class SequentialCoupler(Generic[StateT]):
    """First-order sequential non-iterative reactive-transport model.

    One time step performs conservative transport, local chemistry, geometry
    and property feedback, followed by a flow refresh on the configured
    cadence.  Between flow solves the last velocity field is frozen, while
    reaction and solution-volume mass changes are accumulated conservatively.
    """

    method_name = "first-order multirate SNIA"

    def __init__(
        self,
        *,
        chemistry: ChemistryModel[StateT],
        transport: TransportModel,
        flow: FlowModel,
        dt: float,
        molecular_diffusivity_m2_s: float,
        flow_interval: int,
    ) -> None:
        if dt <= 0.0:
            raise ValueError("coupling time step must be positive")
        if molecular_diffusivity_m2_s < 0.0:
            raise ValueError("molecular diffusivity cannot be negative")
        if flow_interval < 1:
            raise ValueError("flow refresh interval must be at least one")
        self.chemistry = chemistry
        self.transport = transport
        self.flow = flow
        self.dt = float(dt)
        self.molecular_diffusivity_m2_s = float(molecular_diffusivity_m2_s)
        self.flow_interval = int(flow_interval)
        self.flow_solves = 0
        self._flow_reference_bulk_mass: FloatArray | None = None
        self._accumulated_reaction_mass: FloatArray | None = None
        self._elapsed_since_flow = 0.0

    @property
    def has_pending_flow_feedback(self) -> bool:
        return self._elapsed_since_flow > 0.0

    def _solve_flow(
        self,
        chemistry_state: StateT,
        mass_source: FloatArray,
    ) -> tuple[FloatArray, FloatArray, FloatArray, dict[str, float]]:
        u, v, pressure, diagnostics = self.flow.solve(
            chemistry_state.porosity,
            chemistry_state.density_kg_m3,
            chemistry_state.viscosity_pa_s,
            mass_source,
        )
        self.flow_solves += 1
        diagnostics = dict(diagnostics)
        diagnostics["flow_solves"] = self.flow_solves
        return u, v, pressure, diagnostics

    def initialize(self) -> CoupledState[StateT]:
        chemistry_state = self.chemistry.initial_state
        zero_source = np.zeros_like(chemistry_state.porosity, dtype=float)
        u, v, pressure, diagnostics = self._solve_flow(
            chemistry_state, zero_source
        )
        self._flow_reference_bulk_mass = (
            chemistry_state.porosity * chemistry_state.density_kg_m3
        )
        self._accumulated_reaction_mass = np.zeros_like(zero_source)
        self._elapsed_since_flow = 0.0
        return CoupledState(
            time=0.0,
            chemistry=chemistry_state,
            u=u,
            v=v,
            pressure_pa=pressure,
            mass_source_kg_m3_s=zero_source,
            maximum_porosity_change=0.0,
            flow_diagnostics=diagnostics,
        )

    def advance(
        self,
        state: CoupledState[StateT],
        *,
        step_index: int,
        force_flow: bool = False,
    ) -> CoupledState[StateT]:
        if step_index < 1:
            raise ValueError("step_index must start at one")
        if self._flow_reference_bulk_mass is None:
            raise RuntimeError("initialize() must be called before advance()")
        if self._accumulated_reaction_mass is None:
            raise RuntimeError("coupling accumulator is unavailable")

        old_porosity = state.chemistry.porosity
        transported = self.transport.advance(
            state.chemistry.concentrations,
            old_porosity,
            state.u,
            state.v,
            dt=self.dt,
            molecular_diffusivity_m2_s=self.molecular_diffusivity_m2_s,
            inlet_concentrations=self.chemistry.inlet_concentrations,
        )
        chemistry_state = self.chemistry.react(
            transported, old_porosity, self.dt
        )
        maximum_porosity_change = float(
            np.max(np.abs(chemistry_state.porosity - old_porosity))
        )
        self._accumulated_reaction_mass += (
            chemistry_state.solid_mass_release_rate_kg_m3_s * self.dt
        )
        self._elapsed_since_flow += self.dt

        u = state.u
        v = state.v
        pressure = state.pressure_pa
        diagnostics = state.flow_diagnostics
        mass_source = state.mass_source_kg_m3_s
        if force_flow or step_index % self.flow_interval == 0:
            mass_source = self._pending_mass_source(chemistry_state)
            u, v, pressure, diagnostics = self._solve_flow(
                chemistry_state, mass_source
            )
            self._commit_flow_reference(chemistry_state)

        return CoupledState(
            time=state.time + self.dt,
            chemistry=chemistry_state,
            u=u,
            v=v,
            pressure_pa=pressure,
            mass_source_kg_m3_s=mass_source,
            maximum_porosity_change=maximum_porosity_change,
            flow_diagnostics=diagnostics,
        )

    def _pending_mass_source(self, chemistry_state: StateT) -> FloatArray:
        if self._flow_reference_bulk_mass is None:
            raise RuntimeError("flow reference is unavailable")
        if self._accumulated_reaction_mass is None:
            raise RuntimeError("reaction accumulator is unavailable")
        if self._elapsed_since_flow <= 0.0:
            return np.zeros_like(chemistry_state.porosity)
        current_bulk_mass = (
            chemistry_state.porosity * chemistry_state.density_kg_m3
        )
        return (
            self._accumulated_reaction_mass
            - (current_bulk_mass - self._flow_reference_bulk_mass)
        ) / self._elapsed_since_flow

    def _commit_flow_reference(self, chemistry_state: StateT) -> None:
        if self._accumulated_reaction_mass is None:
            raise RuntimeError("reaction accumulator is unavailable")
        self._flow_reference_bulk_mass = (
            chemistry_state.porosity * chemistry_state.density_kg_m3
        ).copy()
        self._accumulated_reaction_mass.fill(0.0)
        self._elapsed_since_flow = 0.0

    def refresh_flow(
        self, state: CoupledState[StateT]
    ) -> CoupledState[StateT]:
        """Apply any pending chemistry/property feedback before stopping."""

        if not self.has_pending_flow_feedback:
            return state
        mass_source = self._pending_mass_source(state.chemistry)
        u, v, pressure, diagnostics = self._solve_flow(
            state.chemistry, mass_source
        )
        self._commit_flow_reference(state.chemistry)
        return CoupledState(
            time=state.time,
            chemistry=state.chemistry,
            u=u,
            v=v,
            pressure_pa=pressure,
            mass_source_kg_m3_s=mass_source,
            maximum_porosity_change=state.maximum_porosity_change,
            flow_diagnostics=diagnostics,
        )
