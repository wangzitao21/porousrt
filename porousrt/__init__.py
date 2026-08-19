"""PorousRT: pore-scale reactive-transport modeling components.

The public API is intentionally expressed in modeling concepts.  Numerical
backends and geochemical engines are implementation details selected by a
case, rather than being encoded in the package name.
"""

from .coupling import (
    ChemistryModel,
    CoupledState,
    FlowModel,
    SequentialCoupler,
    TransportModel,
)
from .artifacts import TimeLevelArchive, TimeLevelRecord
from .flow import (
    VariableDensityFlowSettings,
    VariableDensityFlowSolver,
    solve_variable_density_brinkman_flow,
)
from .boundary_flow import (
    RectangularBoundaryFlowCondition,
    RectangularFlowBoundaryConditions,
    RectangularVariableDensityFlowSettings,
    RectangularVariableDensityFlowSolver,
    solve_rectangular_variable_density_brinkman_flow,
)
from .geometry import build_staggered_square_porosity
from .grid import Grid
from .resources import default_pitzer_database
from .transport import ComponentBoundaryConditions, ConservativeTransportSolver

__all__ = [
    "ChemistryModel",
    "ComponentBoundaryConditions",
    "ConservativeTransportSolver",
    "CoupledState",
    "FlowModel",
    "Grid",
    "RectangularBoundaryFlowCondition",
    "RectangularFlowBoundaryConditions",
    "RectangularVariableDensityFlowSettings",
    "RectangularVariableDensityFlowSolver",
    "SequentialCoupler",
    "TimeLevelArchive",
    "TimeLevelRecord",
    "TransportModel",
    "VariableDensityFlowSettings",
    "VariableDensityFlowSolver",
    "build_staggered_square_porosity",
    "default_pitzer_database",
    "solve_variable_density_brinkman_flow",
    "solve_rectangular_variable_density_brinkman_flow",
]

__version__ = "0.1.0"
