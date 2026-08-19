"""Variable-density Darcy--Brinkman--Stokes flow models."""

from __future__ import annotations

import warnings

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import MatrixRankWarning, spsolve

from dataclasses import dataclass

from .grid import Grid


def _harmonic_scalar(a: float, b: float) -> float:
    return 2.0 * a * b / (a + b + np.finfo(float).tiny)


def _drag_coefficient_nd(
    phi: float,
    viscosity_ratio: float,
    domain_height: float,
    permeability_parameter_m2: float,
    minimum_porosity: float,
    drag_cap_nondimensional: float,
) -> float:
    phi_safe = max(float(phi), minimum_porosity)
    inverse_permeability = (
        domain_height**2
        / permeability_parameter_m2
        * (1.0 - phi_safe) ** 2
        / phi_safe**3
    )
    return viscosity_ratio * min(inverse_permeability, drag_cap_nondimensional)


def solve_variable_density_brinkman_flow(
    porosity: np.ndarray,
    density_kg_m3: np.ndarray,
    viscosity_pa_s: np.ndarray,
    grid: Grid,
    *,
    inlet_velocity_m_s: float,
    inlet_density_kg_m3: float,
    inlet_velocity_profile: np.ndarray | None = None,
    minimum_porosity: float,
    permeability_parameter_m2: float,
    drag_cap_nondimensional: float,
    gravity_m_s2: tuple[float, float] = (0.0, -9.80665),
    mass_balance_source_kg_m3_s: np.ndarray | None = None,
    reference_viscosity_pa_s: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """Solve creeping variable-density Darcy-Brinkman-Stokes flow.

    Momentum uses PHREEQC density and viscosity fields, including reduced
    gravity ``(rho-rho_ref) g``.  The pressure absorbs the constant hydrostatic
    contribution.  Continuity is assembled in conservative mass form,

    ``div(rho * u) = source``.

    Velocity components are on MAC faces and reduced pressure is cell-centred.
    """

    shape = (grid.ny, grid.nx)
    fields = (porosity, density_kg_m3, viscosity_pa_s)
    if any(np.asarray(field).shape != shape for field in fields):
        raise ValueError("flow fields do not match the configured grid")
    if not all(np.all(np.isfinite(field)) for field in fields):
        raise ValueError("flow fields contain non-finite values")
    if np.any(porosity < minimum_porosity) or np.any(porosity > 1.0):
        raise ValueError("porosity is outside configured bounds")
    if np.any(density_kg_m3 <= 0.0) or np.any(viscosity_pa_s <= 0.0):
        raise ValueError("density and viscosity must be positive")
    if inlet_velocity_m_s <= 0.0:
        raise ValueError("inlet velocity must be positive")
    if inlet_density_kg_m3 <= 0.0:
        raise ValueError("inlet density must be positive")
    if permeability_parameter_m2 <= 0.0:
        raise ValueError("permeability parameter must be positive")

    ny, nx = shape
    if inlet_velocity_profile is None:
        inlet_u_nd = np.ones(ny, dtype=float)
    else:
        inlet_u_nd = np.asarray(inlet_velocity_profile, dtype=float).copy()
        if inlet_u_nd.shape != (ny,):
            raise ValueError("inlet velocity profile must have one value per row")
        if not np.all(np.isfinite(inlet_u_nd)) or np.any(inlet_u_nd < 0.0):
            raise ValueError("inlet velocity profile must be finite and nonnegative")
        mean_profile = float(np.mean(inlet_u_nd))
        if mean_profile <= 0.0:
            raise ValueError("inlet velocity profile must contain positive flow")
        inlet_u_nd /= mean_profile
    height = grid.height
    dx = grid.dx / height
    dy = grid.dy / height
    rho_reference = float(
        np.sum(porosity * density_kg_m3) / np.sum(porosity)
    )
    if reference_viscosity_pa_s is None:
        fluid_values = viscosity_pa_s[porosity > 0.9]
        reference_viscosity_pa_s = float(
            np.median(fluid_values if fluid_values.size else viscosity_pa_s)
        )
    if reference_viscosity_pa_s <= 0.0:
        raise ValueError("reference viscosity must be positive")

    if mass_balance_source_kg_m3_s is None:
        mass_source = np.zeros(shape, dtype=float)
    else:
        mass_source = np.asarray(mass_balance_source_kg_m3_s, dtype=float)
        if mass_source.shape != shape or not np.all(np.isfinite(mass_source)):
            raise ValueError("mass-balance source is invalid")
    source_nd = (
        mass_source * height / (rho_reference * inlet_velocity_m_s)
    )

    phi_u = 0.5 * (porosity[:, :-1] + porosity[:, 1:])
    phi_v = 0.5 * (porosity[:-1, :] + porosity[1:, :])
    rho_u_internal = 0.5 * (density_kg_m3[:, :-1] + density_kg_m3[:, 1:])
    rho_v_internal = 0.5 * (density_kg_m3[:-1, :] + density_kg_m3[1:, :])
    mu_u_internal = 2.0 * viscosity_pa_s[:, :-1] * viscosity_pa_s[:, 1:] / (
        viscosity_pa_s[:, :-1]
        + viscosity_pa_s[:, 1:]
        + np.finfo(float).tiny
    )
    mu_v_internal = 2.0 * viscosity_pa_s[:-1, :] * viscosity_pa_s[1:, :] / (
        viscosity_pa_s[:-1, :]
        + viscosity_pa_s[1:, :]
        + np.finfo(float).tiny
    )
    eta_u = (mu_u_internal / reference_viscosity_pa_s) / np.maximum(
        phi_u, minimum_porosity
    )
    eta_v = (mu_v_internal / reference_viscosity_pa_s) / np.maximum(
        phi_v, minimum_porosity
    )

    rho_u = np.empty((ny, nx + 1), dtype=float)
    rho_u[:, 0] = inlet_density_kg_m3
    rho_u[:, -1] = density_kg_m3[:, -1]
    rho_u[:, 1:-1] = rho_u_internal
    rho_v = np.empty((ny + 1, nx), dtype=float)
    rho_v[0, :] = density_kg_m3[0, :]
    rho_v[-1, :] = density_kg_m3[-1, :]
    rho_v[1:-1, :] = rho_v_internal

    inlet_mass_flux_nd = float(
        np.sum((rho_u[:, 0] / rho_reference) * inlet_u_nd) * dy
    )
    integrated_source_nd = float(np.sum(source_nd) * dx * dy)
    outlet_density_integral_nd = float(
        np.sum(rho_u[:, -1] / rho_reference) * dy
    )
    outlet_u_nd = (
        inlet_mass_flux_nd + integrated_source_nd
    ) / outlet_density_integral_nd
    if not np.isfinite(outlet_u_nd) or outlet_u_nd <= 0.0:
        raise RuntimeError(
            "mass-compatible outlet velocity is non-positive; reduce the time step"
        )

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

    def add(row: int, col: int, value: float) -> None:
        rows.append(row)
        cols.append(col)
        data.append(float(value))

    gx, gy = gravity_m_s2
    gravity_scale = height**2 / (
        reference_viscosity_pa_s * inlet_velocity_m_s
    )

    # x-momentum on internal vertical faces.
    for j in range(ny):
        for i in range(1, nx):
            row = iu(j, i)
            eta_here = float(eta_u[j, i - 1])
            mu_ratio = float(mu_u_internal[j, i - 1] / reference_viscosity_pa_s)
            face_phi = float(phi_u[j, i - 1])
            diag = _drag_coefficient_nd(
                face_phi,
                mu_ratio,
                height,
                permeability_parameter_m2,
                minimum_porosity,
                drag_cap_nondimensional,
            )

            if i > 1:
                coeff = _harmonic_scalar(eta_here, float(eta_u[j, i - 2])) / dx**2
                diag += coeff
                add(row, iu(j, i - 1), -coeff)
            else:
                coeff = eta_here / dx**2
                diag += coeff
                rhs[row] += coeff * inlet_u_nd[j]
            if i < nx - 1:
                coeff = _harmonic_scalar(eta_here, float(eta_u[j, i])) / dx**2
                diag += coeff
                add(row, iu(j, i + 1), -coeff)
            else:
                coeff = eta_here / dx**2
                diag += coeff
                rhs[row] += coeff * outlet_u_nd

            if j > 0:
                coeff = _harmonic_scalar(eta_here, float(eta_u[j - 1, i - 1])) / dy**2
                diag += coeff
                add(row, iu(j - 1, i), -coeff)
            else:
                diag += 2.0 * eta_here / dy**2
            if j < ny - 1:
                coeff = _harmonic_scalar(eta_here, float(eta_u[j + 1, i - 1])) / dy**2
                diag += coeff
                add(row, iu(j + 1, i), -coeff)
            else:
                diag += 2.0 * eta_here / dy**2

            add(row, row, diag)
            add(row, ip(j, i), 1.0 / dx)
            add(row, ip(j, i - 1), -1.0 / dx)
            rhs[row] += (
                float(rho_u_internal[j, i - 1]) - rho_reference
            ) * gx * gravity_scale

    # y-momentum on internal horizontal faces.
    for j in range(1, ny):
        for i in range(nx):
            row = iv(j, i)
            eta_here = float(eta_v[j - 1, i])
            mu_ratio = float(mu_v_internal[j - 1, i] / reference_viscosity_pa_s)
            face_phi = float(phi_v[j - 1, i])
            diag = _drag_coefficient_nd(
                face_phi,
                mu_ratio,
                height,
                permeability_parameter_m2,
                minimum_porosity,
                drag_cap_nondimensional,
            )

            if i > 0:
                coeff = _harmonic_scalar(eta_here, float(eta_v[j - 1, i - 1])) / dx**2
                diag += coeff
                add(row, iv(j, i - 1), -coeff)
            else:
                diag += 2.0 * eta_here / dx**2
            if i < nx - 1:
                coeff = _harmonic_scalar(eta_here, float(eta_v[j - 1, i + 1])) / dx**2
                diag += coeff
                add(row, iv(j, i + 1), -coeff)
            else:
                diag += 2.0 * eta_here / dx**2

            if j > 1:
                coeff = _harmonic_scalar(eta_here, float(eta_v[j - 2, i])) / dy**2
                diag += coeff
                add(row, iv(j - 1, i), -coeff)
            else:
                diag += eta_here / dy**2
            if j < ny - 1:
                coeff = _harmonic_scalar(eta_here, float(eta_v[j, i])) / dy**2
                diag += coeff
                add(row, iv(j + 1, i), -coeff)
            else:
                diag += eta_here / dy**2

            add(row, row, diag)
            add(row, ip(j, i), 1.0 / dy)
            add(row, ip(j - 1, i), -1.0 / dy)
            rhs[row] += (
                float(rho_v_internal[j - 1, i]) - rho_reference
            ) * gy * gravity_scale

    # Conservative mass continuity.  One dependent equation is replaced by a
    # reduced-pressure gauge after enforcing global boundary compatibility.
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
                rhs[row] -= (
                    float(rho_u[j, -1] / rho_reference) * outlet_u_nd / dx
                )
            if i > 0:
                add(
                    row,
                    iu(j, i),
                    -float(rho_u[j, i] / rho_reference) / dx,
                )
            else:
                rhs[row] += (
                    float(rho_u[j, 0] / rho_reference) * inlet_u_nd[j] / dx
                )
            if j < ny - 1:
                add(
                    row,
                    iv(j + 1, i),
                    float(rho_v[j + 1, i] / rho_reference) / dy,
                )
            if j > 0:
                add(
                    row,
                    iv(j, i),
                    -float(rho_v[j, i] / rho_reference) / dy,
                )
            rhs[row] += float(source_nd[j, i])

    matrix = coo_matrix((data, (rows, cols)), shape=(total, total)).tocsr()
    with warnings.catch_warnings():
        warnings.simplefilter("error", MatrixRankWarning)
        solution = spsolve(matrix, rhs)
    if not np.all(np.isfinite(solution)):
        raise RuntimeError("variable-density Brinkman solve returned non-finite values")

    u_nd = np.empty((ny, nx + 1), dtype=float)
    u_nd[:, 0] = inlet_u_nd
    u_nd[:, -1] = outlet_u_nd
    for j in range(ny):
        for i in range(1, nx):
            u_nd[j, i] = solution[iu(j, i)]
    v_nd = np.zeros((ny + 1, nx), dtype=float)
    for j in range(1, ny):
        for i in range(nx):
            v_nd[j, i] = solution[iv(j, i)]

    pressure_nd = solution[nu + nv :].reshape(shape)
    u = u_nd * inlet_velocity_m_s
    v = v_nd * inlet_velocity_m_s
    pressure = (
        pressure_nd * reference_viscosity_pa_s * inlet_velocity_m_s / height
    )

    mass_divergence = (
        np.diff(rho_u * u, axis=1) / grid.dx
        + np.diff(rho_v * v, axis=0) / grid.dy
    )
    mass_error = mass_divergence - mass_source
    volume_divergence = (
        np.diff(u, axis=1) / grid.dx + np.diff(v, axis=0) / grid.dy
    )
    uc = 0.5 * (u[:, :-1] + u[:, 1:])
    vc = 0.5 * (v[:-1, :] + v[1:, :])
    speed = np.hypot(uc, vc)
    solid_mask = porosity < 0.1
    solid_speed = speed[solid_mask]
    fluid_speed = speed[~solid_mask]
    mass_scale = rho_reference * inlet_velocity_m_s / height
    diagnostics = {
        "max_mass_balance_error_kg_m3_s": float(np.max(np.abs(mass_error))),
        "rms_mass_balance_error_kg_m3_s": float(
            np.sqrt(np.mean(mass_error**2))
        ),
        "max_relative_mass_balance_error": float(
            np.max(np.abs(mass_error)) / mass_scale
        ),
        "max_volume_divergence_1_s": float(np.max(np.abs(volume_divergence))),
        "outlet_velocity_m_s": float(outlet_u_nd * inlet_velocity_m_s),
        "maximum_inlet_velocity_m_s": float(
            np.max(inlet_u_nd) * inlet_velocity_m_s
        ),
        "max_speed_m_s": float(np.max(speed)),
        "max_vertical_speed_m_s": float(np.max(np.abs(vc))),
        "mean_solid_speed_m_s": float(solid_speed.mean())
        if solid_speed.size
        else 0.0,
        "mean_fluid_speed_m_s": float(fluid_speed.mean())
        if fluid_speed.size
        else 0.0,
        "dynamic_pressure_drop_pa": float(
            np.mean(pressure[:, 0]) - np.mean(pressure[:, -1])
        ),
        "reference_density_kg_m3": rho_reference,
        "reference_viscosity_pa_s": float(reference_viscosity_pa_s),
        "density_contrast_kg_m3": float(
            np.max(density_kg_m3) - np.min(density_kg_m3)
        ),
        "integrated_mass_source_kg_m_s": float(
            np.sum(mass_source) * grid.dx * grid.dy
        ),
        "inlet_mass_flux_kg_m_s": float(
            np.sum(rho_u[:, 0] * u[:, 0]) * grid.dy
        ),
        "outlet_mass_flux_kg_m_s": float(
            np.sum(rho_u[:, -1] * u[:, -1]) * grid.dy
        ),
    }
    return u, v, pressure, diagnostics


@dataclass(frozen=True)
class VariableDensityFlowSettings:
    """Physical and numerical controls for the quasi-steady flow model."""

    inlet_velocity_m_s: float
    inlet_density_kg_m3: float
    inlet_velocity_profile: np.ndarray | None = None
    minimum_porosity: float = 1.0e-3
    permeability_parameter_m2: float = 1.0e-15
    drag_cap_nondimensional: float = 2.0e7
    gravity_m_s2: tuple[float, float] = (0.0, -9.80665)
    reference_viscosity_pa_s: float | None = None


class VariableDensityFlowSolver:
    """Object-oriented flow adapter used by the generic coupler."""

    def __init__(self, grid: Grid, settings: VariableDensityFlowSettings):
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
        return solve_variable_density_brinkman_flow(
            porosity,
            density_kg_m3,
            viscosity_pa_s,
            self.grid,
            inlet_velocity_m_s=settings.inlet_velocity_m_s,
            inlet_density_kg_m3=settings.inlet_density_kg_m3,
            inlet_velocity_profile=settings.inlet_velocity_profile,
            minimum_porosity=settings.minimum_porosity,
            permeability_parameter_m2=settings.permeability_parameter_m2,
            drag_cap_nondimensional=settings.drag_cap_nondimensional,
            gravity_m_s2=settings.gravity_m_s2,
            mass_balance_source_kg_m3_s=mass_balance_source_kg_m3_s,
            reference_viscosity_pa_s=settings.reference_viscosity_pa_s,
        )
