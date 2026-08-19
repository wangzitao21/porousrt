"""Variable-density Brinkman flow with configurable rectangular boundaries."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal
import warnings

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import MatrixRankWarning, spsolve

from .flow import _drag_coefficient_nd, _harmonic_scalar
from .grid import Grid


BoundaryKind = Literal["wall", "velocity", "mass_compatible_outlet"]


@dataclass(frozen=True)
class RectangularBoundaryFlowCondition:
    """Normal-flow condition on one complete side of a rectangular domain.

    ``normal_velocity_m_s`` is positive out of the domain and negative into
    the domain.  An inflow density is required for a prescribed inflow.  A
    ``mass_compatible_outlet`` has its uniform normal velocity calculated on
    every flow refresh so the variable-density continuity equation is globally
    compatible with reaction and solution-volume mass sources.

    All ports use zero prescribed tangential velocity.  This is appropriate
    for normal injection and discharge manifolds and keeps corner conditions
    unambiguous on the staggered grid.
    """

    kind: BoundaryKind = "wall"
    normal_velocity_m_s: float = 0.0
    inflow_density_kg_m3: float | None = None

    def validate(self, side: str) -> None:
        if self.kind not in ("wall", "velocity", "mass_compatible_outlet"):
            raise ValueError(f"unsupported {side} flow boundary kind: {self.kind}")
        if not np.isfinite(self.normal_velocity_m_s):
            raise ValueError(f"{side} boundary velocity must be finite")
        if self.inflow_density_kg_m3 is not None and (
            not np.isfinite(self.inflow_density_kg_m3)
            or self.inflow_density_kg_m3 <= 0.0
        ):
            raise ValueError(f"{side} inflow density must be positive and finite")
        if self.kind != "velocity" and self.normal_velocity_m_s != 0.0:
            raise ValueError(f"{side} {self.kind} boundary cannot prescribe a velocity")
        if (
            self.kind == "velocity"
            and self.normal_velocity_m_s < 0.0
            and self.inflow_density_kg_m3 is None
        ):
            raise ValueError(f"{side} prescribed inflow requires a density")


def _wall() -> RectangularBoundaryFlowCondition:
    return RectangularBoundaryFlowCondition()


@dataclass(frozen=True)
class RectangularFlowBoundaryConditions:
    """Normal-flow conditions for all four sides of a rectangular grid."""

    left: RectangularBoundaryFlowCondition = field(default_factory=_wall)
    right: RectangularBoundaryFlowCondition = field(default_factory=_wall)
    bottom: RectangularBoundaryFlowCondition = field(default_factory=_wall)
    top: RectangularBoundaryFlowCondition = field(default_factory=_wall)

    def items(self) -> tuple[tuple[str, RectangularBoundaryFlowCondition], ...]:
        return (
            ("left", self.left),
            ("right", self.right),
            ("bottom", self.bottom),
            ("top", self.top),
        )

    def validate(self) -> None:
        for side, condition in self.items():
            condition.validate(side)
        compatible_outlets = [
            side
            for side, condition in self.items()
            if condition.kind == "mass_compatible_outlet"
        ]
        if len(compatible_outlets) != 1:
            raise ValueError(
                "exactly one mass-compatible outlet is required; received "
                + str(len(compatible_outlets))
            )
        if not any(
            condition.kind == "velocity" and condition.normal_velocity_m_s < 0.0
            for _, condition in self.items()
        ):
            raise ValueError("at least one prescribed inflow boundary is required")


@dataclass(frozen=True)
class RectangularVariableDensityFlowSettings:
    """Physical, numerical, and boundary controls for rectangular DBS flow."""

    boundaries: RectangularFlowBoundaryConditions
    minimum_porosity: float = 1.0e-3
    permeability_parameter_m2: float = 1.0e-15
    drag_cap_nondimensional: float = 2.0e7
    gravity_m_s2: tuple[float, float] = (0.0, -9.80665)
    reference_viscosity_pa_s: float | None = None
    reference_velocity_m_s: float | None = None


def _adjacent_density(density_kg_m3: np.ndarray, side: str) -> np.ndarray:
    if side == "left":
        return density_kg_m3[:, 0].copy()
    if side == "right":
        return density_kg_m3[:, -1].copy()
    if side == "bottom":
        return density_kg_m3[0, :].copy()
    if side == "top":
        return density_kg_m3[-1, :].copy()
    raise ValueError(f"unknown rectangular boundary side: {side}")


def _boundary_density(
    density_kg_m3: np.ndarray,
    side: str,
    condition: RectangularBoundaryFlowCondition,
) -> np.ndarray:
    adjacent = _adjacent_density(density_kg_m3, side)
    if condition.kind == "velocity" and condition.normal_velocity_m_s < 0.0:
        if condition.inflow_density_kg_m3 is None:  # guarded by validation
            raise ValueError(f"{side} prescribed inflow requires a density")
        return np.full_like(adjacent, condition.inflow_density_kg_m3)
    return adjacent


def solve_rectangular_variable_density_brinkman_flow(
    porosity: np.ndarray,
    density_kg_m3: np.ndarray,
    viscosity_pa_s: np.ndarray,
    grid: Grid,
    *,
    boundaries: RectangularFlowBoundaryConditions,
    minimum_porosity: float,
    permeability_parameter_m2: float,
    drag_cap_nondimensional: float,
    gravity_m_s2: tuple[float, float] = (0.0, -9.80665),
    mass_balance_source_kg_m3_s: np.ndarray | None = None,
    reference_viscosity_pa_s: float | None = None,
    reference_velocity_m_s: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """Solve variable-density DBS flow for configurable rectangular ports.

    The continuity equation is ``div(rho*u) = source``.  Prescribed boundary
    velocities are normal to their side and use the outward-positive sign
    convention.  One boundary must be a mass-compatible outlet; its velocity
    is calculated from all prescribed fluxes and the integrated source.
    """

    boundaries.validate()
    shape = (grid.ny, grid.nx)
    if grid.nx < 2 or grid.ny < 2:
        raise ValueError("rectangular Brinkman flow requires at least 2x2 cells")
    fields = (porosity, density_kg_m3, viscosity_pa_s)
    if any(np.asarray(value).shape != shape for value in fields):
        raise ValueError("flow fields do not match the configured grid")
    if not all(np.all(np.isfinite(value)) for value in fields):
        raise ValueError("flow fields contain non-finite values")
    if not 0.0 < minimum_porosity < 1.0:
        raise ValueError("minimum porosity must lie in (0, 1)")
    if np.any(porosity < minimum_porosity) or np.any(porosity > 1.0):
        raise ValueError("porosity is outside configured bounds")
    if np.any(density_kg_m3 <= 0.0) or np.any(viscosity_pa_s <= 0.0):
        raise ValueError("density and viscosity must be positive")
    if permeability_parameter_m2 <= 0.0 or drag_cap_nondimensional <= 0.0:
        raise ValueError("permeability and drag cap must be positive")

    fixed_speeds = [
        abs(condition.normal_velocity_m_s)
        for _, condition in boundaries.items()
        if condition.kind == "velocity"
    ]
    if reference_velocity_m_s is None:
        reference_velocity_m_s = max(fixed_speeds, default=0.0)
    if not np.isfinite(reference_velocity_m_s) or reference_velocity_m_s <= 0.0:
        raise ValueError("reference velocity must be positive and finite")

    ny, nx = shape
    height = grid.height
    dx = grid.dx / height
    dy = grid.dy / height
    rho_reference = float(np.sum(porosity * density_kg_m3) / np.sum(porosity))
    if reference_viscosity_pa_s is None:
        fluid_values = viscosity_pa_s[porosity > 0.9]
        reference_viscosity_pa_s = float(
            np.median(fluid_values if fluid_values.size else viscosity_pa_s)
        )
    if not np.isfinite(reference_viscosity_pa_s) or reference_viscosity_pa_s <= 0.0:
        raise ValueError("reference viscosity must be positive and finite")

    if mass_balance_source_kg_m3_s is None:
        mass_source = np.zeros(shape, dtype=float)
    else:
        mass_source = np.asarray(mass_balance_source_kg_m3_s, dtype=float)
        if mass_source.shape != shape or not np.all(np.isfinite(mass_source)):
            raise ValueError("mass-balance source is invalid")
    source_nd = mass_source * height / (rho_reference * reference_velocity_m_s)

    boundary_density = {
        side: _boundary_density(density_kg_m3, side, condition)
        for side, condition in boundaries.items()
    }
    normal_velocity_nd: dict[str, float] = {}
    compatible_side = ""
    for side, condition in boundaries.items():
        if condition.kind == "wall":
            normal_velocity_nd[side] = 0.0
        elif condition.kind == "velocity":
            normal_velocity_nd[side] = (
                condition.normal_velocity_m_s / reference_velocity_m_s
            )
        else:
            compatible_side = side

    face_spacing = {
        "left": dy,
        "right": dy,
        "bottom": dx,
        "top": dx,
    }
    known_boundary_flux_nd = 0.0
    for side, normal_velocity in normal_velocity_nd.items():
        known_boundary_flux_nd += float(
            np.sum(boundary_density[side] / rho_reference)
            * normal_velocity
            * face_spacing[side]
        )
    integrated_source_nd = float(np.sum(source_nd) * dx * dy)
    outlet_density_integral_nd = float(
        np.sum(boundary_density[compatible_side] / rho_reference)
        * face_spacing[compatible_side]
    )
    compatible_normal_velocity_nd = (
        integrated_source_nd - known_boundary_flux_nd
    ) / outlet_density_integral_nd
    if (
        not np.isfinite(compatible_normal_velocity_nd)
        or compatible_normal_velocity_nd <= 0.0
    ):
        raise RuntimeError(
            "mass-compatible outlet became an inflow; check boundary fluxes "
            "and reaction time step"
        )
    normal_velocity_nd[compatible_side] = compatible_normal_velocity_nd

    # Convert outward-normal speeds to Cartesian staggered-face components.
    u_left_nd = -normal_velocity_nd["left"]
    u_right_nd = normal_velocity_nd["right"]
    v_bottom_nd = -normal_velocity_nd["bottom"]
    v_top_nd = normal_velocity_nd["top"]

    phi_u = 0.5 * (porosity[:, :-1] + porosity[:, 1:])
    phi_v = 0.5 * (porosity[:-1, :] + porosity[1:, :])
    rho_u_internal = 0.5 * (density_kg_m3[:, :-1] + density_kg_m3[:, 1:])
    rho_v_internal = 0.5 * (density_kg_m3[:-1, :] + density_kg_m3[1:, :])
    mu_u_internal = (
        2.0
        * viscosity_pa_s[:, :-1]
        * viscosity_pa_s[:, 1:]
        / (viscosity_pa_s[:, :-1] + viscosity_pa_s[:, 1:] + np.finfo(float).tiny)
    )
    mu_v_internal = (
        2.0
        * viscosity_pa_s[:-1, :]
        * viscosity_pa_s[1:, :]
        / (viscosity_pa_s[:-1, :] + viscosity_pa_s[1:, :] + np.finfo(float).tiny)
    )
    eta_u = (mu_u_internal / reference_viscosity_pa_s) / np.maximum(
        phi_u, minimum_porosity
    )
    eta_v = (mu_v_internal / reference_viscosity_pa_s) / np.maximum(
        phi_v, minimum_porosity
    )

    rho_u = np.empty((ny, nx + 1), dtype=float)
    rho_u[:, 0] = boundary_density["left"]
    rho_u[:, -1] = boundary_density["right"]
    rho_u[:, 1:-1] = rho_u_internal
    rho_v = np.empty((ny + 1, nx), dtype=float)
    rho_v[0, :] = boundary_density["bottom"]
    rho_v[-1, :] = boundary_density["top"]
    rho_v[1:-1, :] = rho_v_internal

    nu = (nx - 1) * ny
    nv = nx * (ny - 1)
    npres = nx * ny
    total = nu + nv + npres

    def iu(j: int, i: int) -> int:
        return j * (nx - 1) + (i - 1)

    def iv(j: int, i: int) -> int:
        return nu + (j - 1) * nx + i

    def ip(j: int, i: int) -> int:
        return nu + nv + j * nx + i

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    rhs = np.zeros(total, dtype=float)

    def add(row: int, column: int, value: float) -> None:
        rows.append(row)
        cols.append(column)
        data.append(float(value))

    gx, gy = gravity_m_s2
    gravity_scale = height**2 / (reference_viscosity_pa_s * reference_velocity_m_s)

    # x-momentum on internal vertical faces.  Horizontal boundaries prescribe
    # zero tangential velocity, while left/right prescribe their normal speed.
    for j in range(ny):
        for i in range(1, nx):
            row = iu(j, i)
            eta_here = float(eta_u[j, i - 1])
            mu_ratio = float(mu_u_internal[j, i - 1] / reference_viscosity_pa_s)
            face_phi = float(phi_u[j, i - 1])
            diagonal = _drag_coefficient_nd(
                face_phi,
                mu_ratio,
                height,
                permeability_parameter_m2,
                minimum_porosity,
                drag_cap_nondimensional,
            )
            if i > 1:
                coefficient = _harmonic_scalar(eta_here, float(eta_u[j, i - 2])) / dx**2
                diagonal += coefficient
                add(row, iu(j, i - 1), -coefficient)
            else:
                coefficient = eta_here / dx**2
                diagonal += coefficient
                rhs[row] += coefficient * u_left_nd
            if i < nx - 1:
                coefficient = _harmonic_scalar(eta_here, float(eta_u[j, i])) / dx**2
                diagonal += coefficient
                add(row, iu(j, i + 1), -coefficient)
            else:
                coefficient = eta_here / dx**2
                diagonal += coefficient
                rhs[row] += coefficient * u_right_nd
            if j > 0:
                coefficient = (
                    _harmonic_scalar(eta_here, float(eta_u[j - 1, i - 1])) / dy**2
                )
                diagonal += coefficient
                add(row, iu(j - 1, i), -coefficient)
            else:
                diagonal += 2.0 * eta_here / dy**2
            if j < ny - 1:
                coefficient = (
                    _harmonic_scalar(eta_here, float(eta_u[j + 1, i - 1])) / dy**2
                )
                diagonal += coefficient
                add(row, iu(j + 1, i), -coefficient)
            else:
                diagonal += 2.0 * eta_here / dy**2
            add(row, row, diagonal)
            add(row, ip(j, i), 1.0 / dx)
            add(row, ip(j, i - 1), -1.0 / dx)
            rhs[row] += (
                (float(rho_u_internal[j, i - 1]) - rho_reference) * gx * gravity_scale
            )

    # y-momentum on internal horizontal faces.  Bottom/top prescribe their
    # normal speed; left/right prescribe zero tangential velocity.
    for j in range(1, ny):
        for i in range(nx):
            row = iv(j, i)
            eta_here = float(eta_v[j - 1, i])
            mu_ratio = float(mu_v_internal[j - 1, i] / reference_viscosity_pa_s)
            face_phi = float(phi_v[j - 1, i])
            diagonal = _drag_coefficient_nd(
                face_phi,
                mu_ratio,
                height,
                permeability_parameter_m2,
                minimum_porosity,
                drag_cap_nondimensional,
            )
            if i > 0:
                coefficient = (
                    _harmonic_scalar(eta_here, float(eta_v[j - 1, i - 1])) / dx**2
                )
                diagonal += coefficient
                add(row, iv(j, i - 1), -coefficient)
            else:
                diagonal += 2.0 * eta_here / dx**2
            if i < nx - 1:
                coefficient = (
                    _harmonic_scalar(eta_here, float(eta_v[j - 1, i + 1])) / dx**2
                )
                diagonal += coefficient
                add(row, iv(j, i + 1), -coefficient)
            else:
                diagonal += 2.0 * eta_here / dx**2
            if j > 1:
                coefficient = _harmonic_scalar(eta_here, float(eta_v[j - 2, i])) / dy**2
                diagonal += coefficient
                add(row, iv(j - 1, i), -coefficient)
            else:
                coefficient = eta_here / dy**2
                diagonal += coefficient
                rhs[row] += coefficient * v_bottom_nd
            if j < ny - 1:
                coefficient = _harmonic_scalar(eta_here, float(eta_v[j, i])) / dy**2
                diagonal += coefficient
                add(row, iv(j + 1, i), -coefficient)
            else:
                coefficient = eta_here / dy**2
                diagonal += coefficient
                rhs[row] += coefficient * v_top_nd
            add(row, row, diagonal)
            add(row, ip(j, i), 1.0 / dy)
            add(row, ip(j - 1, i), -1.0 / dy)
            rhs[row] += (
                (float(rho_v_internal[j - 1, i]) - rho_reference) * gy * gravity_scale
            )

    # Conservative mass continuity.  Compatibility makes one cell equation
    # dependent, so it is replaced with a reduced-pressure gauge.
    gauge_j, gauge_i = ny - 1, nx - 1
    for j in range(ny):
        for i in range(nx):
            row = ip(j, i)
            if (j, i) == (gauge_j, gauge_i):
                add(row, ip(j, i), 1.0)
                continue
            if i < nx - 1:
                add(
                    row,
                    iu(j, i + 1),
                    float(rho_u[j, i + 1] / rho_reference) / dx,
                )
            else:
                rhs[row] -= float(rho_u[j, -1] / rho_reference) * u_right_nd / dx
            if i > 0:
                add(
                    row,
                    iu(j, i),
                    -float(rho_u[j, i] / rho_reference) / dx,
                )
            else:
                rhs[row] += float(rho_u[j, 0] / rho_reference) * u_left_nd / dx
            if j < ny - 1:
                add(
                    row,
                    iv(j + 1, i),
                    float(rho_v[j + 1, i] / rho_reference) / dy,
                )
            else:
                rhs[row] -= float(rho_v[-1, i] / rho_reference) * v_top_nd / dy
            if j > 0:
                add(
                    row,
                    iv(j, i),
                    -float(rho_v[j, i] / rho_reference) / dy,
                )
            else:
                rhs[row] += float(rho_v[0, i] / rho_reference) * v_bottom_nd / dy
            rhs[row] += float(source_nd[j, i])

    matrix = coo_matrix((data, (rows, cols)), shape=(total, total)).tocsr()
    with warnings.catch_warnings():
        warnings.simplefilter("error", MatrixRankWarning)
        solution = spsolve(matrix, rhs)
    if not np.all(np.isfinite(solution)):
        raise RuntimeError("rectangular Brinkman solve returned non-finite values")

    u_nd = np.empty((ny, nx + 1), dtype=float)
    u_nd[:, 0] = u_left_nd
    u_nd[:, -1] = u_right_nd
    for j in range(ny):
        for i in range(1, nx):
            u_nd[j, i] = solution[iu(j, i)]
    v_nd = np.empty((ny + 1, nx), dtype=float)
    v_nd[0, :] = v_bottom_nd
    v_nd[-1, :] = v_top_nd
    for j in range(1, ny):
        for i in range(nx):
            v_nd[j, i] = solution[iv(j, i)]

    pressure_nd = solution[nu + nv :].reshape(shape)
    u = u_nd * reference_velocity_m_s
    v = v_nd * reference_velocity_m_s
    pressure = pressure_nd * reference_viscosity_pa_s * reference_velocity_m_s / height

    mass_divergence = (
        np.diff(rho_u * u, axis=1) / grid.dx + np.diff(rho_v * v, axis=0) / grid.dy
    )
    mass_error = mass_divergence - mass_source
    volume_divergence = np.diff(u, axis=1) / grid.dx + np.diff(v, axis=0) / grid.dy
    uc = 0.5 * (u[:, :-1] + u[:, 1:])
    vc = 0.5 * (v[:-1, :] + v[1:, :])
    speed = np.hypot(uc, vc)
    solid_mask = porosity < 0.1
    solid_speed = speed[solid_mask]
    fluid_speed = speed[~solid_mask]
    mass_scale = rho_reference * reference_velocity_m_s / height
    outward_mass_fluxes = {
        "left": float(np.sum(rho_u[:, 0] * -u[:, 0]) * grid.dy),
        "right": float(np.sum(rho_u[:, -1] * u[:, -1]) * grid.dy),
        "bottom": float(np.sum(rho_v[0, :] * -v[0, :]) * grid.dx),
        "top": float(np.sum(rho_v[-1, :] * v[-1, :]) * grid.dx),
    }
    diagnostics = {
        "max_mass_balance_error_kg_m3_s": float(np.max(np.abs(mass_error))),
        "rms_mass_balance_error_kg_m3_s": float(np.sqrt(np.mean(mass_error**2))),
        "max_relative_mass_balance_error": float(
            np.max(np.abs(mass_error)) / mass_scale
        ),
        "max_volume_divergence_1_s": float(np.max(np.abs(volume_divergence))),
        "max_speed_m_s": float(np.max(speed)),
        "max_vertical_speed_m_s": float(np.max(np.abs(vc))),
        "mean_solid_speed_m_s": float(solid_speed.mean()) if solid_speed.size else 0.0,
        "mean_fluid_speed_m_s": float(fluid_speed.mean()) if fluid_speed.size else 0.0,
        "pressure_range_pa": float(np.max(pressure) - np.min(pressure)),
        "reference_density_kg_m3": rho_reference,
        "reference_viscosity_pa_s": float(reference_viscosity_pa_s),
        "reference_velocity_m_s": float(reference_velocity_m_s),
        "density_contrast_kg_m3": float(np.max(density_kg_m3) - np.min(density_kg_m3)),
        "integrated_mass_source_kg_m_s": float(np.sum(mass_source) * grid.dx * grid.dy),
        "net_outward_boundary_mass_flux_kg_m_s": float(
            sum(outward_mass_fluxes.values())
        ),
        "mass_compatible_outlet_normal_velocity_m_s": float(
            compatible_normal_velocity_nd * reference_velocity_m_s
        ),
    }
    for side, value in outward_mass_fluxes.items():
        diagnostics[f"{side}_outward_mass_flux_kg_m_s"] = value
        diagnostics[f"{side}_outward_normal_velocity_m_s"] = float(
            normal_velocity_nd[side] * reference_velocity_m_s
        )
    return u, v, pressure, diagnostics


class RectangularVariableDensityFlowSolver:
    """Flow adapter for :class:`porousrt.coupling.SequentialCoupler`."""

    def __init__(
        self, grid: Grid, settings: RectangularVariableDensityFlowSettings
    ) -> None:
        settings.boundaries.validate()
        self.grid = grid
        self.settings = settings

    def solve(
        self,
        porosity: np.ndarray,
        density_kg_m3: np.ndarray,
        viscosity_pa_s: np.ndarray,
        mass_balance_source_kg_m3_s: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
        settings = self.settings
        return solve_rectangular_variable_density_brinkman_flow(
            porosity,
            density_kg_m3,
            viscosity_pa_s,
            self.grid,
            boundaries=settings.boundaries,
            minimum_porosity=settings.minimum_porosity,
            permeability_parameter_m2=settings.permeability_parameter_m2,
            drag_cap_nondimensional=settings.drag_cap_nondimensional,
            gravity_m_s2=settings.gravity_m_s2,
            mass_balance_source_kg_m3_s=mass_balance_source_kg_m3_s,
            reference_viscosity_pa_s=settings.reference_viscosity_pa_s,
            reference_velocity_m_s=settings.reference_velocity_m_s,
        )
