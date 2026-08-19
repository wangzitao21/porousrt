"""Conservative multicomponent transport backend for reactive models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .grid import Grid


BOUNDARY_SIDES = ("left", "right", "bottom", "top")


@dataclass(frozen=True)
class ComponentBoundaryConditions:
    """Component concentrations prescribed on rectangular inflow boundaries.

    Each non-``None`` value is a complete PHREEQC component vector.  The
    transport solver applies Dirichlet values only on the sides declared as
    ``inlet_sides`` when it is constructed.  Other sides retain the natural
    zero-diffusive-flux condition unless they are explicitly listed as
    outflows.
    """

    left: np.ndarray | None = None
    right: np.ndarray | None = None
    bottom: np.ndarray | None = None
    top: np.ndarray | None = None

    def value(self, side: str) -> np.ndarray | None:
        if side not in BOUNDARY_SIDES:
            raise ValueError(f"unknown rectangular boundary side: {side}")
        return getattr(self, side)

    @property
    def active_sides(self) -> tuple[str, ...]:
        return tuple(side for side in BOUNDARY_SIDES if self.value(side) is not None)


class ConservativeTransportSolver:
    """Advance all aqueous master components on a shared finite-volume mesh.

    The equation solved for every component is

    ``d(phi C)/dt + div(u C) = div(phi**2 D grad(C))``.

    Porosity and velocity are frozen over one SNIA transport substep.  A
    first-order upwind convection term is deliberately used for boundedness in
    concentrated brines.  Every component uses the same effective diffusivity;
    this keeps the transported charge balance internally consistent without
    pretending to implement a full electro-diffusion model.
    """

    backend_name = "FiPy"

    def __init__(
        self,
        grid: Grid,
        *,
        inlet_sides: tuple[str, ...] = ("left",),
        outlet_sides: tuple[str, ...] = ("right",),
    ):
        try:
            from fipy import (
                CellVariable,
                DiffusionTerm,
                FaceVariable,
                Grid2D,
                TransientTerm,
                UpwindConvectionTerm,
            )
            from fipy.solvers.scipy import LinearLUSolver
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "FiPy is required for the carnallite replacement example; "
                "run it in the configured 'work' Python environment"
            ) from exc

        invalid_sides = (set(inlet_sides) | set(outlet_sides)) - set(BOUNDARY_SIDES)
        if invalid_sides:
            raise ValueError(
                "unknown rectangular boundary side(s): "
                + ", ".join(sorted(invalid_sides))
            )
        if not inlet_sides:
            raise ValueError("at least one transport inlet side is required")
        if len(set(inlet_sides)) != len(inlet_sides):
            raise ValueError("transport inlet sides must be unique")
        if len(set(outlet_sides)) != len(outlet_sides):
            raise ValueError("transport outlet sides must be unique")
        overlap = set(inlet_sides) & set(outlet_sides)
        if overlap:
            raise ValueError(
                "a transport side cannot be both inlet and outlet: "
                + ", ".join(sorted(overlap))
            )

        self.grid = grid
        self.inlet_sides = tuple(inlet_sides)
        self.outlet_sides = tuple(outlet_sides)
        self.mesh = Grid2D(
            nx=grid.nx,
            ny=grid.ny,
            dx=grid.dx,
            dy=grid.dy,
        )
        self._variable = CellVariable(
            mesh=self.mesh, value=0.0, hasOld=True, name="component_total"
        )
        self._porosity = CellVariable(mesh=self.mesh, value=1.0, name="porosity")
        self._gamma_cell = CellVariable(
            mesh=self.mesh, value=0.0, name="effective_diffusivity"
        )
        self._gamma_face = FaceVariable(mesh=self.mesh, value=0.0)
        self._velocity = FaceVariable(mesh=self.mesh, rank=1, value=(0.0, 0.0))
        self._equation = (
            TransientTerm(coeff=self._porosity, var=self._variable)
            + UpwindConvectionTerm(coeff=self._velocity, var=self._variable)
            == DiffusionTerm(coeff=self._gamma_face, var=self._variable)
        )
        self._solver = LinearLUSolver()
        side_faces = {
            "left": self.mesh.facesLeft,
            "right": self.mesh.facesRight,
            "bottom": self.mesh.facesBottom,
            "top": self.mesh.facesTop,
        }
        self._inlet_face_values: dict[str, FaceVariable] = {}
        for side in self.inlet_sides:
            face_value = FaceVariable(mesh=self.mesh, value=0.0)
            self._inlet_face_values[side] = face_value
            self._variable.constrain(face_value, where=side_faces[side])
        for side in self.outlet_sides:
            self._variable.faceGrad.constrain(
                ((0.0,), (0.0,)), where=side_faces[side]
            )

        centers = np.asarray(self.mesh.faceCenters, dtype=float)
        normals = np.asarray(self.mesh.faceNormals, dtype=float)
        self._vertical_faces = np.abs(normals[0]) > 0.5
        self._horizontal_faces = np.abs(normals[1]) > 0.5

        self._vertical_i = np.clip(
            np.rint(centers[0, self._vertical_faces] / grid.dx).astype(int),
            0,
            grid.nx,
        )
        self._vertical_j = np.clip(
            np.floor(centers[1, self._vertical_faces] / grid.dy).astype(int),
            0,
            grid.ny - 1,
        )
        self._horizontal_i = np.clip(
            np.floor(centers[0, self._horizontal_faces] / grid.dx).astype(int),
            0,
            grid.nx - 1,
        )
        self._horizontal_j = np.clip(
            np.rint(centers[1, self._horizontal_faces] / grid.dy).astype(int),
            0,
            grid.ny,
        )

    def _set_face_velocity(self, u: np.ndarray, v: np.ndarray) -> None:
        values = np.zeros((2, self.mesh.numberOfFaces), dtype=float)
        values[0, self._vertical_faces] = u[
            self._vertical_j, self._vertical_i
        ]
        values[1, self._horizontal_faces] = v[
            self._horizontal_j, self._horizontal_i
        ]
        self._velocity.setValue(values)

    def advance(
        self,
        concentrations: np.ndarray,
        porosity: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        *,
        dt: float,
        molecular_diffusivity_m2_s: float,
        inlet_concentrations: np.ndarray | ComponentBoundaryConditions,
    ) -> np.ndarray:
        """Advance one transport substep and return ``(ncomp, ny, nx)``."""

        values = np.asarray(concentrations, dtype=float)
        phi = np.asarray(porosity, dtype=float)
        if isinstance(inlet_concentrations, ComponentBoundaryConditions):
            unexpected = set(inlet_concentrations.active_sides) - set(
                self.inlet_sides
            )
            if unexpected:
                raise ValueError(
                    "component values were supplied for non-inlet side(s): "
                    + ", ".join(sorted(unexpected))
                )
            inlet_values: dict[str, np.ndarray] = {}
            for side in self.inlet_sides:
                side_value = inlet_concentrations.value(side)
                if side_value is None:
                    raise ValueError(
                        f"component concentrations are missing for {side} inlet"
                    )
                inlet_values[side] = np.asarray(side_value, dtype=float)
        else:
            shared_inlet = np.asarray(inlet_concentrations, dtype=float)
            inlet_values = {
                side: shared_inlet for side in self.inlet_sides
            }
        expected = (self.grid.ny, self.grid.nx)
        if values.ndim != 3 or values.shape[1:] != expected:
            raise ValueError("concentrations must have shape (components, ny, nx)")
        if phi.shape != expected:
            raise ValueError("porosity shape does not match the FiPy mesh")
        if u.shape != (self.grid.ny, self.grid.nx + 1):
            raise ValueError("u does not use the expected MAC layout")
        if v.shape != (self.grid.ny + 1, self.grid.nx):
            raise ValueError("v does not use the expected MAC layout")
        for side, inlet in inlet_values.items():
            if inlet.shape != (values.shape[0],):
                raise ValueError(
                    f"{side} inlet component vector has the wrong shape"
                )
        if dt <= 0.0 or molecular_diffusivity_m2_s < 0.0:
            raise ValueError("transport time step and diffusivity are invalid")
        if np.any(phi <= 0.0) or np.any(phi > 1.0):
            raise ValueError("porosity must lie in (0, 1]")
        if not all(
            np.all(np.isfinite(field))
            for field in (values, phi, u, v, *inlet_values.values())
        ):
            raise ValueError("FiPy transport inputs contain non-finite values")

        self._porosity.setValue(phi.reshape(-1))
        self._gamma_cell.setValue(
            (phi**2 * molecular_diffusivity_m2_s).reshape(-1)
        )
        self._gamma_face.setValue(self._gamma_cell.harmonicFaceValue)
        self._set_face_velocity(u, v)

        updated = np.empty_like(values)
        for component in range(values.shape[0]):
            self._variable.setValue(values[component].reshape(-1))
            self._variable.updateOld()
            for side, inlet in inlet_values.items():
                self._inlet_face_values[side].setValue(float(inlet[component]))
            self._equation.solve(
                var=self._variable,
                dt=float(dt),
                solver=self._solver,
            )
            updated[component] = np.asarray(
                self._variable.value, dtype=float
            ).reshape(expected)

        if not np.all(np.isfinite(updated)):
            raise RuntimeError("FiPy returned non-finite component concentrations")
        return updated
