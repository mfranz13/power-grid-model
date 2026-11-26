# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Fork extension: hybrid AC/DC power flow behavior not covered by the data-driven validation
cases: guards, batch/update equivalence, status switching and control-configuration errors.
Numerical correctness is anchored by tests/data/power_flow/hybrid-dc-* (verified against the
independent acdcpf.generalized reference)."""

import numpy as np
import pytest

from power_grid_model import (
    CalculationInitialization,
    CalculationMethod,
    LoadGenType,
    PowerGridModel,
    initialize_array,
)
from power_grid_model.enum import VscControlMode, VscQControlMode
from power_grid_model.errors import PowerGridBatchError
from power_grid_model.validation import validate_input_data
from power_grid_model.validation.errors import MissingValueError, NotGreaterOrEqualError

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


def test_converter_flow_in_bus_balance():
    input_data = p2p_input()
    result = PowerGridModel(input_data).calculate_power_flow(**HYBRID)
    line, vsc = input_data["line"], input_data["vsc"]
    for quantity in ("p", "q"):
        # the source at node 1 feeds the line and the converter attached there
        branch_out = (
            result["line"][f"{quantity}_from"][line["from_node"] == 1].sum()
            + result["line"][f"{quantity}_to"][line["to_node"] == 1].sum()
            + result["vsc"][f"{quantity}_from"][vsc["from_node"] == 1].sum()
        )
        assert result["source"][quantity][0] == pytest.approx(branch_out, abs=1e-2)
        # node output at both converter buses is the sum of its appliance injections
        for node_id in (1, 2):
            appliance_injection = (
                result["source"][quantity][input_data["source"]["node"] == node_id].sum()
                - result["sym_load"][quantity][input_data["sym_load"]["node"] == node_id].sum()
            )
            node_value = result["node"][quantity][result["node"]["id"] == node_id][0]
            assert node_value == pytest.approx(appliance_injection, abs=1e-2)


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
    assert result["dc_node"]["u_pu"][1, 1] == pytest.approx(1.0, abs=1e-2)  # still energized through the cable
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


def assert_loss_equation(result, input_data, k):
    vsc = input_data["vsc"]
    i_ac = result["vsc"]["i_from"][k]
    loss = vsc["loss_a"][k] + vsc["loss_b"][k] * i_ac + vsc["loss_c"][k] * i_ac**2
    # tolerance: 1e-8 pu of the 1 MVA base
    assert result["vsc"]["p_from"][k] + result["vsc"]["p_to"][k] == pytest.approx(loss, abs=1e-2)


def u_ac_input(u_ac_pu=1.02):
    input_data = p2p_input()
    input_data["vsc"]["control_q"] = [VscQControlMode.q, VscQControlMode.u_ac]
    input_data["vsc"]["q_set"] = [0.5e6, np.nan]
    input_data["vsc"]["u_ac_set"] = [np.nan, u_ac_pu * U_AC]
    return input_data


def test_u_ac_control_p2p():
    input_data = u_ac_input()
    assert validate_input_data(input_data, calculation_type=None) is None
    result = PowerGridModel(input_data).calculate_power_flow(**HYBRID)
    assert result["node"]["u_pu"][1] == pytest.approx(1.02, abs=1e-8)
    assert result["vsc"]["p_from"][1] == pytest.approx(-2e6, abs=1e-3)
    for k in range(2):
        assert_loss_equation(result, input_data, k)
    # the converter must deliver reactive power to lift its bus above the source voltage
    assert result["vsc"]["q_from"][1] < 0.0
    dc_line_loss = result["dc_line"]["p_from"][0] + result["dc_line"]["p_to"][0]
    assert result["vsc"]["p_to"].sum() + dc_line_loss == pytest.approx(0.0, abs=1.0)


def test_u_ac_combined_with_u_dc():
    input_data = u_ac_input(u_ac_pu=0.99)
    input_data["vsc"]["control_p"] = [VscControlMode.p, VscControlMode.u_dc]
    input_data["vsc"]["p_set"] = [1.5e6, np.nan]
    input_data["vsc"]["u_dc_set"] = [np.nan, 1.01 * U_DC]
    result = PowerGridModel(input_data).calculate_power_flow(**HYBRID)
    assert result["node"]["u_pu"][1] == pytest.approx(0.99, abs=1e-8)
    assert result["dc_node"]["u_pu"][1] == pytest.approx(1.01, abs=1e-8)
    assert result["vsc"]["p_from"][0] == pytest.approx(1.5e6, abs=1e-3)
    for k in range(2):
        assert_loss_equation(result, input_data, k)


def test_double_u_ac_pin_raises():
    input_data = u_ac_input()
    input_data["vsc"]["from_node"] = [2, 2]
    input_data["vsc"]["control_q"] = [VscQControlMode.u_ac, VscQControlMode.u_ac]
    input_data["vsc"]["u_ac_set"] = [U_AC, U_AC]
    with pytest.raises(RuntimeError, match="more than one u_ac-controlling"):
        PowerGridModel(input_data).calculate_power_flow(**HYBRID)


def test_u_ac_at_source_bus_raises():
    input_data = u_ac_input()
    input_data["vsc"]["control_q"] = [VscQControlMode.u_ac, VscQControlMode.q]
    input_data["vsc"]["q_set"] = [np.nan, 0.0]
    input_data["vsc"]["u_ac_set"] = [U_AC, np.nan]
    with pytest.raises(RuntimeError, match=r"u_ac-controlling converter at AC bus .* with a source"):
        PowerGridModel(input_data).calculate_power_flow(**HYBRID)


def test_u_ac_missing_setpoint_validation():
    input_data = u_ac_input()
    input_data["vsc"]["u_ac_set"] = np.nan
    errors = validate_input_data(input_data, calculation_type=None)
    assert errors is not None
    assert [(type(e), e.field, e.ids) for e in errors] == [(MissingValueError, "u_ac_set", [14])]


def add_regulated_gen(input_data, node_id, q_min=-5e6, q_max=5e6, u_ref=1.03):
    """Add AC node 15 (line 16 from node 2) and a regulated sym_gen at node_id."""
    node = initialize_array("input", "node", 1)
    node["id"], node["u_rated"] = [15], [U_AC]
    line = initialize_array("input", "line", 1)
    line["id"], line["from_node"], line["to_node"] = [16], [2], [15]
    line["from_status"] = line["to_status"] = [1]
    line["r1"], line["x1"], line["c1"], line["tan1"] = 0.3, 0.8, 0.0, 0.0
    line["r0"], line["x0"], line["c0"], line["tan0"] = 0.6, 1.6, 0.0, 0.0
    gen = initialize_array("input", "sym_gen", 1)
    gen["id"], gen["node"], gen["status"] = [7], [node_id], [1]
    gen["type"] = [LoadGenType.const_power]
    gen["p_specified"], gen["q_specified"] = [0.5e6], [0.0]
    regulator = initialize_array("input", "voltage_regulator", 1)
    regulator["id"], regulator["regulated_object"], regulator["status"] = [8], [7], [1]
    regulator["u_ref"], regulator["q_min"], regulator["q_max"] = [u_ref], [q_min], [q_max]
    input_data["node"] = np.concatenate([input_data["node"], node])
    input_data["line"] = np.concatenate([input_data["line"], line])
    input_data["sym_gen"] = gen
    input_data["voltage_regulator"] = regulator
    return input_data


@pytest.mark.parametrize(("q_max", "limit_hit"), [(5e6, False), (0.1e6, True)])
def test_hybrid_matches_newton_raphson_with_voltage_regulator(q_max, limit_hit):
    ac_only = {k: v for k, v in p2p_input().items() if k in ("node", "line", "source", "sym_load")}
    model = PowerGridModel(add_regulated_gen(ac_only, node_id=15, q_max=q_max))
    nr = model.calculate_power_flow(symmetric=True, calculation_method=CalculationMethod.newton_raphson)
    hybrid = model.calculate_power_flow(**HYBRID)
    assert (nr["voltage_regulator"]["limit_violated"][0] != 0) == limit_hit
    for component in ("node", "line", "sym_gen", "voltage_regulator"):
        for field in nr[component].dtype.names:
            np.testing.assert_array_equal(hybrid[component][field], nr[component][field])


@pytest.mark.parametrize("with_regulator", [False, True])
@pytest.mark.parametrize("initialization", [CalculationInitialization.flat, CalculationInitialization.average_source])
def test_hybrid_matches_newton_raphson_with_initialization(initialization, with_regulator):
    ac_only = {k: v for k, v in p2p_input().items() if k in ("node", "line", "source", "sym_load")}
    if with_regulator:
        ac_only = add_regulated_gen(ac_only, node_id=15)
    model = PowerGridModel(ac_only)
    settings = HYBRID | {"calculation_initialization": initialization}
    nr = model.calculate_power_flow(**(settings | {"calculation_method": CalculationMethod.newton_raphson}))
    hybrid = model.calculate_power_flow(**settings)
    for component in nr:
        for field in nr[component].dtype.names:
            np.testing.assert_array_equal(hybrid[component][field], nr[component][field])


@pytest.mark.parametrize("make_input", [p2p_input, u_ac_input])
@pytest.mark.parametrize("initialization", [CalculationInitialization.flat, CalculationInitialization.average_source])
def test_dc_initialization_matches_linear_start(initialization, make_input):
    model = PowerGridModel(make_input())
    linear = model.calculate_power_flow(**HYBRID)
    other = model.calculate_power_flow(**(HYBRID | {"calculation_initialization": initialization}))
    # powers compared in per unit of the 1 MVA base
    for component, field, scale in (
        ("node", "u_pu", 1.0),
        ("dc_node", "u_pu", 1.0),
        ("vsc", "p_from", 1e6),
        ("vsc", "q_from", 1e6),
        ("vsc", "p_to", 1e6),
    ):
        np.testing.assert_allclose(other[component][field] / scale, linear[component][field] / scale, rtol=0, atol=1e-8)


def test_voltage_regulator_with_dc_link():
    input_data = add_regulated_gen(p2p_input(), node_id=15)
    result = PowerGridModel(input_data).calculate_power_flow(**HYBRID)
    assert result["node"]["u_pu"][2] == pytest.approx(1.03, abs=1e-8)
    assert result["voltage_regulator"]["limit_violated"][0] == 0
    for k in range(2):
        assert_loss_equation(result, input_data, k)
    dc_line_loss = result["dc_line"]["p_from"][0] + result["dc_line"]["p_to"][0]
    assert result["vsc"]["p_to"].sum() + dc_line_loss == pytest.approx(0.0, abs=1.0)


def test_voltage_regulator_at_q_controlled_converter_bus():
    input_data = add_regulated_gen(p2p_input(), node_id=2)
    input_data["vsc"]["q_set"] = [0.5e6, 0.3e6]
    result = PowerGridModel(input_data).calculate_power_flow(**HYBRID)
    assert result["node"]["u_pu"][1] == pytest.approx(1.03, abs=1e-8)
    assert result["vsc"]["q_from"][1] == pytest.approx(0.3e6, abs=1e-3)
    # bus Q balance at node 2: the regulated generator also covers the converter's reactive draw
    branch_q_out = result["line"]["q_to"][0] + result["line"]["q_from"][1] + result["vsc"]["q_from"][1]
    assert result["sym_gen"]["q"][0] - result["sym_load"]["q"][0] == pytest.approx(branch_q_out, abs=1e-2)


def test_voltage_regulator_at_u_ac_converter_bus_raises():
    input_data = add_regulated_gen(u_ac_input(), node_id=2)
    with pytest.raises(RuntimeError, match="voltage regulator and u_ac"):
        PowerGridModel(input_data).calculate_power_flow(**HYBRID)


@pytest.mark.parametrize(("q_min", "q_max"), [(-10e6, 10e6), (np.nan, 10e6), (np.nan, np.nan)])
def test_u_ac_inactive_q_limits_match_unlimited(q_min, q_max):
    unlimited = PowerGridModel(u_ac_input()).calculate_power_flow(**HYBRID)
    input_data = u_ac_input()
    input_data["vsc"]["q_min"] = [np.nan, q_min]
    input_data["vsc"]["q_max"] = [np.nan, q_max]
    limited = PowerGridModel(input_data).calculate_power_flow(**HYBRID)
    if np.isnan(q_min) and np.isnan(q_max):
        np.testing.assert_array_equal(result_vector(limited), result_vector(unlimited))
    else:
        # the armed limit check adds an iteration, so the result is equal only to solver tolerance
        np.testing.assert_allclose(result_vector(limited) / 1e6, result_vector(unlimited) / 1e6, rtol=0, atol=1e-10)
    assert limited["vsc"]["q_from"][1] == pytest.approx(unlimited["vsc"]["q_from"][1], abs=1e-3)


# unlimited Q_t at u_ac_set 0.98 pu is about +2.4 Mvar, at 1.04 pu about -4.3 Mvar
@pytest.mark.parametrize(
    ("u_ac_pu", "q_min", "q_max", "q_clamped"),
    [(0.98, np.nan, 1e6, 1e6), (1.04, -2e6, np.nan, -2e6)],
)
def test_u_ac_q_limit_clamps_and_releases_voltage(u_ac_pu, q_min, q_max, q_clamped):
    input_data = u_ac_input(u_ac_pu=u_ac_pu)
    input_data["vsc"]["q_min"] = [np.nan, q_min]
    input_data["vsc"]["q_max"] = [np.nan, q_max]
    # limits of a q-mode converter are ignored: its q_set lies above this q_max
    input_data["vsc"]["q_max"][0] = 0.1e6
    assert validate_input_data(input_data, calculation_type=None) is None
    result = PowerGridModel(input_data).calculate_power_flow(**HYBRID)
    assert result["vsc"]["q_from"][1] == pytest.approx(q_clamped, abs=1e-6)
    assert result["vsc"]["q_from"][0] == pytest.approx(0.5e6, abs=1e-6)
    # at q_max the converter absorbs less than needed, so the voltage ends above the setpoint (below at q_min)
    assert np.sign(result["node"]["u_pu"][1] - u_ac_pu) == np.sign(q_clamped)
    assert result["node"]["u_pu"][1] != pytest.approx(u_ac_pu, abs=1e-3)
    for k in range(2):
        assert_loss_equation(result, input_data, k)


def test_u_ac_q_limit_order_validation():
    input_data = u_ac_input()
    input_data["vsc"]["q_min"] = [np.nan, 1e6]
    input_data["vsc"]["q_max"] = [np.nan, -1e6]
    errors = validate_input_data(input_data, calculation_type=None)
    assert errors == [NotGreaterOrEqualError("vsc", "q_max", [14], "q_min")]
