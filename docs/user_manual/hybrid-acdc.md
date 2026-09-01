<!--
SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>

SPDX-License-Identifier: MPL-2.0
-->

# Hybrid AC/DC power flow (fork extension)

This fork adds a unified hybrid AC/DC Newton-Raphson power flow based on the generalized
formulation of Zulkifli et al. (IEEE TPWRS 2026): one combined Newton system over the AC grid, the
DC grid and the converters, solved by the same sparse block-LU machinery as the plain AC
Newton-Raphson solver. It is not part of upstream power-grid-model.

## Components

### `dc_node`

A DC bus. Derives from `node`; `u_rated` is the rated DC voltage (pole-to-pole). Output is the
standard node output with `u_angle = 0` and `q = 0`. No source, load, generator, shunt, sensor or
link may be connected to a `dc_node`.

| name | data type | unit | description | required |
| --- | --- | --- | --- | :---: |
| `u_rated` | `double` | volt (V) | rated DC voltage | &#10004; |

### `dc_line`

A DC cable between two `dc_node`s of equal `u_rated`. Enters the admittance matrix as the purely
real series admittance `1 / r1`; `r1` is the total loop resistance. Standard branch output; the
current base has no `sqrt(3)` factor (`i = p / u_dc`).

| name | data type | unit | description | required |
| --- | --- | --- | --- | :---: |
| `r1` | `double` | ohm (Ω) | series resistance (> 0) | &#10004; |
| `i_n` | `double` | ampere (A) | rated current (for `loading`) | |

### `vsc`

An AC/DC voltage source converter, modeled as a terminal black box: AC-side powers `P_t`/`Q_t`
and DC-side power `P_f`, coupled by the IEC 62751-2 loss equation
`P_f + P_t = loss_a + loss_b · I_t + loss_c · I_t²` with `I_t = |S_t| / U_ac`.
The `from` side must be an AC `node`, the `to` side a `dc_node`. Setpoints use the from-side
branch convention: positive `p_set`/`q_set` means power flowing from the AC node into the
converter. The reactive power `q_set` is always a fixed setpoint.

| name | data type | unit | description | required |
| --- | --- | --- | --- | :---: |
| `control_p` | `VscControlMode` | - | `p` (0), `u_dc` (1) or `droop` (2) | &#10004; |
| `p_set` | `double` | watt (W) | AC-side active power setpoint | mode `p` |
| `q_set` | `double` | var | AC-side reactive power setpoint | &#10004; |
| `u_dc_set` | `double` | volt (V) | DC voltage setpoint | mode `u_dc` |
| `droop_p_set` | `double` | watt (W) | droop base power | mode `droop` |
| `droop_gain` | `double` | watt/volt (W/V) | droop gain (> 0): `P_t = droop_p_set + droop_gain · (droop_u_ref − U_dc)` | mode `droop` |
| `droop_u_ref` | `double` | volt (V) | droop reference voltage (default: `u_rated` of the `dc_node`) | |
| `loss_a` | `double` | watt (W) | constant loss term | &#10004; |
| `loss_b` | `double` | volt (V) | linear loss term (× AC terminal current) | &#10004; |
| `loss_c` | `double` | ohm (Ω) | quadratic loss term (× AC terminal current²) | &#10004; |
| `sn` | `double` | VA | rated power (for `loading`) | |

Output is the standard branch output: `p_from`/`q_from` = AC-side powers `P_t`/`Q_t`,
`p_to` = DC-side power `P_f` (power flowing from the DC bus into the converter), so
`p_from + p_to` is the converter loss. All attributes except `r1` of `dc_line` are updatable;
setpoint/mode updates do not invalidate any cached topology or admittance data, which makes DC
time-series batches as cheap as load time series.

## Calculation method

Select `CalculationMethod.hybrid_newton_raphson` (value 101). Grids containing DC components only
support the symmetric power flow calculation with this method; any other calculation type,
method or the asymmetric calculation is rejected. The method also works on grids without DC
components and then reproduces the plain Newton-Raphson results exactly.

Every galvanically connected DC subnetwork needs at least one converter in `u_dc` or `droop`
mode (the voltage anchor), and at most one `u_dc` converter per `dc_node`. Violations raise a
diagnostic error.

Restrictions of this first version: VSC converters only (no LCC), symmetric calculation only, no
voltage regulators together with DC components, converter reactive power is always a fixed
setpoint (no AC voltage control by the converter).

## Algorithm

The converter unknowns are eliminated symbolically before the Newton iteration, so the hybrid
system stays a pure bus-level 2×2-block system on the standard `YBusStructure` sparsity pattern:

- `P_f` is eliminated through the loss equation (linear in `P_f`).
- In `droop` mode, `P_t` is eliminated through the droop equation (linear in `P_t` and `U_dc`).
- In `u_dc` mode, the pinned DC voltage frees the V-column of that bus; `P_t` takes the slot.
- DC buses replace their Q-equation by the identity `Δθ = 0`; with a real admittance block the
  generic H/N/M/L formulas then produce exactly the DC power-balance derivatives.

Every VSC is registered as a zero-admittance branch, which reserves the required off-diagonal
sparsity slots and merges AC and DC into one math model. The elimination is exact, so the
quadratic convergence of Newton-Raphson is preserved; typical cases converge in 4 iterations. The
implementation is verified against the independent `acdcpf.generalized` reference (jax automatic
differentiation) at agreement levels of 1e-11 to 1e-13.

## Performance (mv_oberrhein, flattened, 185 AC buses, SOP DC link)

| case | warm median | notes |
| --- | --- | --- |
| AC only, `newton_raphson` | 0.27 ms | baseline |
| AC only, `hybrid_newton_raphson` | 0.28 ms | +4.6 %, results bit-identical to `newton_raphson` |
| AC + SOP link, `hybrid_newton_raphson` | 0.34 ms | 2 extra buses, 3 extra branches, converter assembly |
| same case, unified jax reference | ≈ 1 700–2 000 ms | dense AD Jacobian (hybrid_loadflow `docs/timing_acdc.md`) |
| same case, sequential PGM-AC + numpy-DC | ≈ 4.6–5.1 ms | outer AC/DC loop |

Batch: 1000 VSC-setpoint scenarios in ≈ 5 ms total (≈ 5 µs/scenario) on a small test grid —
topology, admittance matrix and solver instances stay cached across DC parameter updates.
