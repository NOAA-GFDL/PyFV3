# TEMPORARY BRANCH: Up-skilling to GEOS v11.4.2

This branch exists solely for up-skilling pyFV3 to be able run GEOS in it's v11.4.2 FP configuration.
The need for a seperate branch from `develop` rely in the following differences:

- GEOS run a 32bit floating point precision version (with appropriate 64bit buffers for mass conservation). This means the translate test requires a new set of data _and_ will not pass on old 8.1.3 Pace data.
- GEOS requires options that are deemed "legacy" and that we may want to replace rather than port
- Project requirements demand quick iterative development, while `pyFV3` demands concertation between all stakeholders.

The aim is to validate and benchmark GEOS v11.4.2 with this dynamics. Once done, we will aim to move _as much code as possible_ back into develop.
The methodology goes as follows

- Merge directly into `develop` any changes that do not demand a new set of data
- Keep track of the feature branch (below) that can't be merged in `develop` for future PR
- Keep track of GEOS vs SHiELD differences for future discussions

## Feature branches

Legend:

- ⚙️ _GEOS - WIP_ : Ongoing work - can be merged temporarily
- 🔶 _GEOS - Merged_:  Considered done - merged in GEOS v11.4.2 branch but NOT in `develop`
- ✅ _Develop - Merged_: Work done as part of up-skilling done for GEOS merged in `develop` AND the GEOS v11.4.2 branch.

Branches:

- ✅ `fix/F32/UpdateDzC`@Florian: Fix for fluxes gradient
- ✅ `fix/F32/DivergenceDamping`@Florian: Fix for 32-bit scalars in DivergenceDamping
- ✅ `fix/F32/UpdateDzD`@Florian: Fix for fluxes gradient & python computation
- ✅ `fix/F32/nh_p_grad` @ Florian: Fix for 32-bit NonHydrostaticPressureGradient
- 🔶 `fix/RayleighDamping_mixed_precision`@Florian: fix the Ray_Fast test
- 🔶 `GEOS_update/yppm_xppm`@Florian: fix the YPPM/XPPM with `hord = -6`
- 🔶 `fix/DelnFlux_f32_support`@Florian: Fix for f32 support for DelnFlux (partial pass)
- ⚙️ `fix/GEOS/D_SW`@Florian: Fix D_SW heat dissipation, column calculation and new `dpx` accumulation (partial pass)
- ⚙️ `fix/GEOSv11_4_2/A2B_Ord4`@Florian: Fix for 32-bit A2B_Ord4
- ⚙️ `fix/GEOSv11_4_2/RiemanSolver`@Florian: Fix for 32-bit A2B_Ord4
- ⚙️ `fix/GEOSv11_4_2/C_SW`@Florian: Fix for C_SW for 32-bit
- ⚙️ `fix/GEOSv11_4_2/Dyncore`@Florian: Fix for Acoustics and DycoreState for 32-bit and `dpx` calculation
  - MERGE ORDER: after `fix/GEOS/D_SW`
- ⚙️ `feature/tracer_rework_part1` @Florian: Allow for update of N Tracers
- ⚙️ `fix/GEOS/TracerAdvection` @Florian: Allow for non-update of mass fluxes and courant number, f32 fixes, correct computation of `cmax` and `nsplit`, overcomputation into the algorithm
  - BASED ON `tracer_rework_part1`
  - REQUIRES: `ndsl` with tracer rework
- ⚙️ `feature/fv_mapz/GEOS` @ Chris K: Remapping for GEOS
