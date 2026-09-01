# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Fork extension: hybrid AC/DC power flow behavior not covered by the data-driven validation
cases: guards, batch/update equivalence, status switching and control-configuration errors.
Numerical correctness is anchored by tests/data/power_flow/hybrid-dc-* (verified against the
independent acdcpf.generalized reference)."""

import numpy as np
import pytest

from power_grid_model import CalculationMethod, LoadGenType, PowerGridModel, initialize_array
from power_grid_model.enum import VscControlMode
from power_grid_model.errors import PowerGridBatchError

U_AC = 10.5e3
U_DC = 20e3
HYBRID = {
    "symmetric": True,
    "calculation_method": CalculationMethod.hybrid_newton_raphson,
    "error_tolerance": 1e-10,
    "max_iterations": 30,
}


def p2p_input(p_set_2=-2e6, u_dc_set_1=U_DC, q_set_1=0.5e6):
    node = initialize_array("input", "node", 2)
    node["id"] = [1, 2]
    node["u_rated"] = U_AC
    line = initialize_array("input", "line", 1)
    line["id"], line["from_node"], line["to_node"] = [3], [1], [2]
    line["from_status"] = line["to_status"] = [1]
    line["r1"], line["x1"], line["c1"], line["tan1"] = 0.5, 1.0, 0.0, 0.0
    line["r0"], line["x0"], line["c0"], line["tan0"] = 1.0, 2.0, 0.0, 0.0
    source = initialize_array("input", "source", 1)
    source["id"], source["node"], source["status"], source["u_ref"] = [5], [1], [1], [1.0]
    load = initialize_array("input", "sym_load", 1)
    load["id"], load["node"], load["status"] = [6], [2], [1]
    load["type"] = [LoadGenType.const_power]
    load["p_specified"], load["q_specified"] = [1e6], [0.2e6]
    dc_node = initialize_array("input", "dc_node", 2)
    dc_node["id"] = [10, 11]
    dc_node["u_rated"] = U_DC
    dc_line = initialize_array("input", "dc_line", 1)
    dc_line["id"], dc_line["from_node"], dc_line["to_node"] = [12], [10], [11]
    dc_line["from_status"] = dc_line["to_status"] = [1]
    dc_line["r1"] = [2.0]
    vsc = initialize_array("input", "vsc", 2)
    vsc["id"] = [13, 14]
    vsc["from_node"], vsc["to_node"] = [1, 2], [10, 11]
    vsc["from_status"] = vsc["to_status"] = [1, 1]
    vsc["control_p"] = [VscControlMode.u_dc, VscControlMode.p]
    vsc["p_set"] = [np.nan, p_set_2]
    vsc["q_set"] = [q_set_1, 0.0]
    vsc["u_dc_set"] = [u_dc_set_1, np.nan]
    vsc["loss_a"], vsc["loss_b"], vsc["loss_c"] = 5e3, 0.02, 1.0
    return {
        "node": node,
        "line": line,
        "source": source,
        "sym_load": load,
        "dc_node": dc_node,
        "dc_line": dc_line,
        "vsc": vsc,
    }


def result_vector(result, scenario=None):
    idx = (scenario, slice(None)) if scenario is not None else slice(None)
    return np.concatenate(
        [
            result["node"]["u_pu"][idx],
            result["dc_node"]["u_pu"][idx],
            result["vsc"]["p_from"][idx],
            result["vsc"]["p_to"][idx],
        ]
    )


def test_dc_requires_hybrid_method():
    model = PowerGridModel(p2p_input())
    with pytest.raises(RuntimeError, match="hybrid_newton_raphson"):
        model.calculate_power_flow(symmetric=True, calculation_method=CalculationMethod.newton_raphson)
    with pytest.raises(RuntimeError, match="hybrid_newton_raphson"):
        model.calculate_power_flow(symmetric=False, calculation_method=CalculationMethod.hybrid_newton_raphson)
    with pytest.raises(RuntimeError):
        model.calculate_state_estimation(symmetric=True)


def test_hybrid_method_matches_newton_raphson_on_pure_ac():
    ac_only = {k: v for k, v in p2p_input().items() if k in ("node", "line", "source", "sym_load")}
    model = PowerGridModel(ac_only)
    u_nr = model.calculate_power_flow(symmetric=True, calculation_method=CalculationMethod.newton_raphson)["node"][
        "u_pu"
    ]
    u_hybrid = model.calculate_power_flow(**HYBRID)["node"]["u_pu"]
    np.testing.assert_array_equal(u_hybrid, u_nr)


def test_solution_invariants():
    result = PowerGridModel(p2p_input()).calculate_power_flow(**HYBRID)
    # pinned DC voltage and p setpoint are met exactly
    assert result["dc_node"]["u"][0] == pytest.approx(U_DC, abs=1e-6)
    assert result["vsc"]["p_from"][1] == pytest.approx(-2e6, abs=1e-3)
    # DC power balance: converter DC-side powers plus cable loss sum to zero
    dc_line_loss = result["dc_line"]["p_from"][0] + result["dc_line"]["p_to"][0]
    assert result["vsc"]["p_to"].sum() + dc_line_loss == pytest.approx(0.0, abs=1.0)


def test_batch_setpoint_sweep_equals_fresh_models():
    n = 10
    p_sweep = np.linspace(-3e6, 2.5e6, n)
    update = initialize_array("update", "vsc", (n, 1))
    update["id"] = 14
    update["p_set"][:, 0] = p_sweep
    batch = PowerGridModel(p2p_input()).calculate_power_flow(update_data={"vsc": update}, **HYBRID)
    for scenario in (0, n // 2, n - 1):
        fresh = PowerGridModel(p2p_input(p_set_2=p_sweep[scenario])).calculate_power_flow(**HYBRID)
        np.testing.assert_allclose(result_vector(batch, scenario), result_vector(fresh), rtol=0, atol=1e-10)


def test_permanent_update_equals_fresh_model():
    model = PowerGridModel(p2p_input())
    update = initialize_array("update", "vsc", 1)
    update["id"], update["p_set"] = [14], [-1.1e6]
    model.update(update_data={"vsc": update})
    fresh = PowerGridModel(p2p_input(p_set_2=-1.1e6)).calculate_power_flow(**HYBRID)
    np.testing.assert_allclose(result_vector(model.calculate_power_flow(**HYBRID)), result_vector(fresh), atol=1e-10)


def test_status_switching_batch():
    # scenario 0: all on; 1: vsc2 off (dc2 fed via cable, zero transfer); 2: dead dc2 island
    vsc_update = initialize_array("update", "vsc", (3, 1))
    vsc_update["id"] = 14
    vsc_update["from_status"] = [[1], [0], [0]]
    vsc_update["to_status"] = [[1], [0], [0]]
    dcl_update = initialize_array("update", "dc_line", (3, 1))
    dcl_update["id"] = 12
    dcl_update["from_status"] = [[1], [1], [0]]
    result = PowerGridModel(p2p_input()).calculate_power_flow(
        update_data={"vsc": vsc_update, "dc_line": dcl_update}, **HYBRID
    )
    assert result["vsc"]["energized"][1, 1] == 0
    assert result["vsc"]["p_to"][1, 0] == pytest.approx(0.0, abs=1e4)  # only losses remain
    assert result["dc_node"]["u_pu"][1, 1] > 0.99  # still energized through the cable
    assert result["dc_node"]["energized"][2, 1] == 0  # dead island
    assert result["dc_node"]["energized"][2, 0] == 1


def test_missing_voltage_anchor_raises():
    input_data = p2p_input()
    input_data["vsc"]["control_p"] = [VscControlMode.p, VscControlMode.p]
    input_data["vsc"]["p_set"] = [2e6, -2e6]
    model = PowerGridModel(input_data)
    with pytest.raises(RuntimeError, match="u_dc- or droop-controlling"):
        model.calculate_power_flow(**HYBRID)


def test_double_udc_pin_raises():
    input_data = p2p_input()
    # move both converters onto one dc_node, both in u_dc mode
    input_data["vsc"]["to_node"] = [10, 10]
    input_data["vsc"]["control_p"] = [VscControlMode.u_dc, VscControlMode.u_dc]
    input_data["vsc"]["u_dc_set"] = [U_DC, U_DC]
    model = PowerGridModel(input_data)
    with pytest.raises(RuntimeError, match="more than one u_dc-controlling"):
        model.calculate_power_flow(**HYBRID)


def test_batch_error_isolated_per_scenario():
    update = initialize_array("update", "vsc", (2, 1))
    update["id"] = 13  # the only anchor
    update["from_status"] = [[1], [0]]
    update["to_status"] = [[1], [1]]
    model = PowerGridModel(p2p_input())
    with pytest.raises(PowerGridBatchError) as exc_info:
        model.calculate_power_flow(update_data={"vsc": update}, **HYBRID)
    assert list(exc_info.value.failed_scenarios) == [1]
