# PorousRT architecture

`porousrt` is the reusable pore-scale reactive-transport modeling package.
Backend names are deliberately absent from its public modeling API.

```text
cases/<case>/run.py
        |
        v
case chemistry + diagnostics + visualization
        |
        v
porousrt.SequentialCoupler
   |          |          |
transport   chemistry   variable-density flow
   |          |          |
implicit FV  local      Darcy--Brinkman--Stokes
components   reactions  + geometry/property feedback
```

## Dependency direction

- `porousrt/` never imports `cases/`.
- A case supplies a chemistry object satisfying `ChemistryModel` and owns all
  mineral-specific phase definitions, parameters, plots, and stop criteria.
- `SequentialCoupler` depends only on the `TransportModel`, `ChemistryModel`,
  and `FlowModel` protocols.
- `grid.py`, `geometry.py`, and `media.py` provide small shared utilities;
  neither case imports an earlier monolithic simulation script.
- `flow.py` preserves the original left-inlet/right-outlet solver.
  `boundary_flow.py` adds outward-normal ports on all rectangular sides and a
  mass-compatible outlet without changing the legacy API.
- `ComponentBoundaryConditions` lets one transport instance prescribe a
  different component vector on each configured inflow side.  Passing the
  original one-dimensional inlet vector still applies the legacy behavior.
- `TimeLevelArchive` is a case-independent writer for a compressed full-field
  NPZ and refreshed JSON manifest at every time level.
- Generated artifacts stay under `cases/<case>/results/`.  The framework and
  tests never write to the protected `ref/` or `outputs/` directories.

## One SNIA step

1. Advance every aqueous master component conservatively with frozen porosity
   and velocity.
2. Apply local chemistry to the transported component totals.
3. Convert mineral changes to porosity and a net solid-to-fluid mass source.
4. Update density and viscosity from the chemistry state.
5. Refresh the quasi-steady flow at its configured cadence.  Accumulated
   reaction and solution-volume changes enter conservative mass continuity.

`refresh_flow()` applies pending feedback before an early-stop frame, so a
dissolved obstacle cannot leave behind a stale velocity solution.

## Adding a case

Create `cases/<name>/` with `chemistry.py`, `model.py`, `run.py`, `README.md`,
and `__main__.py`.  Implement the small chemistry state contract, compose the
three core models with `SequentialCoupler`, and keep case-specific diagnostics
outside the framework package.  Add a chemistry unit test and a short coupled
smoke run before a full production calculation.

Closely related cases may expose a thin local `chemistry.py` that reuses an
already-tested mineral adapter while supplying different reaction mode,
solution, geometry, diagnostics, and output configuration.  The kinetic
wormholing case and its all-equilibrium sibling follow this pattern for the
Carnallite/Sylvite/Halite system.

For multiple streams, construct `ConservativeTransportSolver` with the desired
`inlet_sides`/`outlet_sides`, return `ComponentBoundaryConditions` from the
chemistry adapter, and use `RectangularVariableDensityFlowSolver` with exactly
one mass-compatible outlet.
