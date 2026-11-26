# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Anschauliches Beispiel zum hybriden AC/DC-Lastfluss: zwei asynchrone 110-kV-Inseln,
per 33-kV-DC-Link auf der 20-kV-Ebene gekoppelt. Zeigt die Betriebsmodi p / u_dc / droop
und die Wirkung der Setpoints. Ausführen mit:  uv run python examples/hybrid_dc_link.py
(optional mit --with matplotlib für die Abbildung hybrid_dc_link.png)

    Slack A                            Slack B
    node 1 (110 kV)                    node 2 (110 kV)
      |                                  |
    Trafo 21 (110/20, 10 MVA)          Trafo 22
      |                                  |
    node 3 (20 kV) ==[VSC 71]== dc 61 --dc_line 63 (33 kV)-- dc 62 ==[VSC 72]== node 4 (20 kV)
      |                                  |
    Leitung 31                         Leitung 32
      |                                  |
    node 5: Last 800 kW                node 6: Erzeugung 500 kW
"""

import numpy as np
from power_grid_model import (
    BranchSide,
    CalculationMethod,
    LoadGenType,
    PowerGridModel,
    WindingType,
    initialize_array,
)
from power_grid_model.enum import VscControlMode

U_HV = 110e3
U_LV = 20e3
U_DC = 33e3
P_LAST = 800e3
P_ERZEUGUNG = 500e3

# Modulationsindexschätzung nach Song: U_s = K_MOD * M * U_dc mit K_MOD = sqrt(6)/4.
# Das Terminal-Modell (Zulkifli) kennt kein M — diese Prüfung ist eine reine
# Plausibilitätsanzeige: M > 1 heißt, der Konverter könnte die AC-Spannung physisch
# nicht mehr stellen (Übermodulation).
K_MOD = np.sqrt(6.0) / 4.0

HYBRID = {
    "symmetric": True,
    "calculation_method": CalculationMethod.hybrid_newton_raphson,
    "error_tolerance": 1e-10,
    "max_iterations": 30,
}


def build_input(control=("u_dc", "p"), u_dc_set=U_DC, p_set_2=300e3, droop=None):
    """Netz aus dem Kopfdiagramm. `control` sind die P-Modi der VSC (71, 72);
    `droop = (p_set_1, gain_1, p_set_2, gain_2)` parametriert den droop-Modus [W, W/V]."""
    node = initialize_array("input", "node", 6)
    node["id"] = [1, 2, 3, 4, 5, 6]
    node["u_rated"] = [U_HV, U_HV, U_LV, U_LV, U_LV, U_LV]

    # pandapower std_type "25 MVA 110/20 kV": vk 12 %, vkr 0.41 %, pfe 14 kW, i0 0.07 %,
    # YNd5, Stufensteller HV-seitig 1.5 %/Stufe (-9..+9)
    trafo = initialize_array("input", "transformer", 2)
    trafo["id"] = [21, 22]
    trafo["from_node"], trafo["to_node"] = [1, 2], [3, 4]
    trafo["from_status"] = trafo["to_status"] = 1
    trafo["u1"], trafo["u2"] = U_HV, U_LV
    trafo["sn"] = 25e6
    trafo["uk"], trafo["pk"] = 0.12, 0.0041 * 25e6
    trafo["i0"], trafo["p0"] = 0.0007, 14e3
    trafo["winding_from"] = WindingType.wye_n
    trafo["winding_to"] = WindingType.delta
    trafo["clock"] = 5
    trafo["tap_side"] = BranchSide.from_side
    trafo["tap_pos"] = trafo["tap_nom"] = 0
    trafo["tap_min"], trafo["tap_max"] = -9, 9
    trafo["tap_size"] = 0.015 * U_HV

    # pandapower std_type "NA2XS2Y 1x240 RM/25 12/20 kV" (0.122 + j0.112 Ohm/km,
    # 304 nF/km, 421 A), Laenge 5 km; Nullsystem fuer die symmetrische Rechnung irrelevant
    line_length_km = 5.0
    line = initialize_array("input", "line", 2)
    line["id"] = [31, 32]
    line["from_node"], line["to_node"] = [3, 4], [5, 6]
    line["from_status"] = line["to_status"] = 1
    line["r1"], line["x1"] = 0.122 * line_length_km, 0.112 * line_length_km
    line["c1"], line["tan1"] = 304e-9 * line_length_km, 0.0
    line["r0"], line["x0"], line["c0"], line["tan0"] = 0.4 * line_length_km, 0.35 * line_length_km, 0.0, 0.0
    line["i_n"] = 421.0

    source = initialize_array("input", "source", 2)
    source["id"], source["node"], source["status"] = [41, 42], [1, 2], 1
    source["u_ref"] = 1.0

    sym_load = initialize_array("input", "sym_load", 1)
    sym_load["id"], sym_load["node"], sym_load["status"] = [51], [5], [1]
    sym_load["type"] = LoadGenType.const_power
    sym_load["p_specified"], sym_load["q_specified"] = P_LAST, 200e3

    sym_gen = initialize_array("input", "sym_gen", 1)
    sym_gen["id"], sym_gen["node"], sym_gen["status"] = [52], [6], [1]
    sym_gen["type"] = LoadGenType.const_power
    sym_gen["p_specified"], sym_gen["q_specified"] = P_ERZEUGUNG, 0.0

    dc_node = initialize_array("input", "dc_node", 2)
    dc_node["id"] = [61, 62]
    dc_node["u_rated"] = U_DC

    # DC-Link mit demselben AC-Kabeltyp (NA2XS2Y 1x240 RM/25): Hin- und Rueckleiter,
    # Schleifenwiderstand r1 = 2 * 0.122 Ohm/km * Laenge
    dc_length_km = 2.0
    dc_line = initialize_array("input", "dc_line", 1)
    dc_line["id"] = [63]
    dc_line["from_node"], dc_line["to_node"] = [61], [62]
    dc_line["from_status"] = dc_line["to_status"] = 1
    dc_line["r1"] = 2.0 * 0.122 * dc_length_km
    dc_line["i_n"] = 421.0

    vsc = initialize_array("input", "vsc", 2)
    vsc["id"] = [71, 72]
    vsc["from_node"], vsc["to_node"] = [3, 4], [61, 62]
    vsc["from_status"] = vsc["to_status"] = 1
    modes = {"p": VscControlMode.p, "u_dc": VscControlMode.u_dc, "droop": VscControlMode.droop}
    vsc["control_p"] = [modes[control[0]], modes[control[1]]]
    vsc["u_dc_set"] = [u_dc_set if control[0] == "u_dc" else np.nan,
                       u_dc_set if control[1] == "u_dc" else np.nan]
    vsc["p_set"] = [np.nan, p_set_2 if control[1] == "p" else np.nan]
    vsc["q_set"] = [-100e3, 0.0]  # VSC 71 liefert 100 kvar in die Last-Insel
    if droop is not None:
        p1, g1, p2, g2 = droop
        vsc["droop_p_set"] = [p1, p2]
        vsc["droop_gain"] = [g1, g2]
        vsc["droop_u_ref"] = U_DC
    vsc["loss_a"], vsc["loss_b"], vsc["loss_c"] = 8e3, 0.0, 0.5

    return {
        "node": node, "transformer": trafo, "line": line, "source": source,
        "sym_load": sym_load, "sym_gen": sym_gen,
        "dc_node": dc_node, "dc_line": dc_line, "vsc": vsc,
    }


def modulation_index(result, scenario=None):
    """M je Konverter aus AC-Klemmenspannung (Knoten 3/4) und lokaler DC-Spannung."""
    idx = (scenario, slice(None)) if scenario is not None else slice(None)
    u_ac = result["node"]["u"][idx][..., [2, 3]]
    u_dc = result["dc_node"]["u"][idx]
    return u_ac / (K_MOD * u_dc)


def print_scenario(result, scenario=None, label=""):
    idx = (scenario, slice(None)) if scenario is not None else slice(None)
    p_t = result["vsc"]["p_from"][idx] / 1e3
    q_t = result["vsc"]["q_from"][idx] / 1e3
    p_f = result["vsc"]["p_to"][idx] / 1e3
    loss = p_t + p_f
    u_dc = result["dc_node"]["u"][idx] / 1e3
    m = modulation_index(result, scenario)
    p_src = result["source"]["p"][idx] / 1e3
    loading = result["transformer"]["loading"][idx] * 100
    transfer = -p_f[0]  # Leistung, die am DC-Knoten 61 ankommt (B -> A positiv)
    print(f"  {label}")
    print(f"    Übertragung DC-Link (B->A): {transfer:8.1f} kW   U_dc = {u_dc[0]:.3f} / {u_dc[1]:.3f} kV")
    for k, name in enumerate(("VSC 71 (Insel A)", "VSC 72 (Insel B)")):
        flag = "  ** M > 1: Übermodulation! **" if m[k] > 1.0 else ""
        print(f"    {name}: P_t = {p_t[k]:8.1f} kW  Q_t = {q_t[k]:7.1f} kvar  "
              f"P_f = {p_f[k]:8.1f} kW  Verlust = {loss[k]:5.1f} kW  M = {m[k]:.3f}{flag}")
    print(f"    Slack A: {p_src[0]:8.1f} kW   Slack B: {p_src[1]:8.1f} kW   "
          f"Trafo-Auslastung: {loading[0]:.1f} % / {loading[1]:.1f} %")


def scenario_a():
    print("\n=== A: Modus p + u_dc — der P-Setpoint diktiert die Übertragung ===")
    print("VSC 72 im p-Modus (Setpoint = Aufnahme aus Insel B), VSC 71 hält U_dc = 33 kV")
    print("und übernimmt als Bilanzknoten automatisch Übertragung + Verluste.\n")
    p_sets = np.array([-600e3, -300e3, 0.0, 300e3, 600e3])
    update = initialize_array("update", "vsc", (p_sets.size, 1))
    update["id"] = 72
    update["p_set"][:, 0] = p_sets
    model = PowerGridModel(build_input())
    result = model.calculate_power_flow(update_data={"vsc": update}, **HYBRID)
    for s, p in enumerate(p_sets):
        print_scenario(result, s, f"p_set(VSC 72) = {p / 1e3:+.0f} kW")
    print("\n  Lesart: p_set > 0 zieht Leistung aus der Erzeugungs-Insel B zur Last-Insel A")
    print("  (Slack A wird entlastet), p_set < 0 kehrt die Richtung um. Der u_dc-Konverter")
    print("  folgt spiegelbildlich, versetzt um die Konverter- und Kabelverluste.")
    return p_sets, result


def scenario_b():
    print("\n=== B: u_dc-Setpoint — Spannungslage des DC-Links und Modulationsindex ===")
    print("Gleiche Übertragung (300 kW), aber U_dc-Sollwert variiert. Die Leistungsflüsse")
    print("bleiben praktisch identisch — aber M = U_AC / (K_MOD * U_dc) reagiert:\n")
    u_sets = np.array([31e3, 32e3, 33e3, 35e3])
    update = initialize_array("update", "vsc", (u_sets.size, 1))
    update["id"] = 71
    update["u_dc_set"][:, 0] = u_sets
    model = PowerGridModel(build_input())
    result = model.calculate_power_flow(update_data={"vsc": update}, **HYBRID)
    for s, u in enumerate(u_sets):
        print_scenario(result, s, f"u_dc_set = {u / 1e3:.0f} kV")
    print("\n  Lesart: 33 kV ist für 20 kV AC gerade ausreichend (M ≈ 0.99); darunter")
    print("  meldet die Plausibilitätsprüfung Übermodulation. Das Terminal-Modell rechnet")
    print("  trotzdem weiter — die M-Grenze ist eine physikalische Randbedingung, die der")
    print("  Anwender (oder das physikalische Song-Modell) prüfen muss.")
    return u_sets, result


def scenario_c():
    print("\n=== C: Droop beidseitig — Fahrplan-Abweichung wird über U_dc aufgeteilt ===")
    print("Beide VSC im droop-Modus: P_t = droop_p_set + gain * (33 kV - U_dc).")
    print("VSC 71 plant -400 kW (Lieferung an A), der Fahrplan von VSC 72 wird variiert.")
    print("Bilanzieren die Fahrpläne nicht, verschiebt sich U_dc, bis die Gains die")
    print("Differenz schließen — aufgeteilt im Verhältnis der Gains (hier 100 : 50 kW/kV).\n")
    gains = (100.0, 50.0)  # W/V = kW/kV: 100 kW/kV und 50 kW/kV
    p2_sets = np.array([300e3, 400e3, 500e3, 600e3])
    update = initialize_array("update", "vsc", (p2_sets.size, 1))
    update["id"] = 72
    update["droop_p_set"][:, 0] = p2_sets
    model = PowerGridModel(build_input(
        control=("droop", "droop"), droop=(-400e3, gains[0], 300e3, gains[1])))
    result = model.calculate_power_flow(update_data={"vsc": update}, **HYBRID)
    for s, p2 in enumerate(p2_sets):
        mismatch = (-400e3 + p2) / 1e3
        print_scenario(result, s, f"Fahrplan: VSC71 = -400 kW, VSC72 = {p2 / 1e3:+.0f} kW  "
                                  f"(Überschuss {mismatch:+.0f} kW)")
    print("\n  Lesart: Je 100 kW Fahrplan-Überschuss verschieben U_dc um ~0.56 kV")
    print("  (= 100 kW - Verluständerung, geteilt durch die Gain-Summe 150 kW/kV); VSC 71")
    print("  weicht wegen des doppelten Gains doppelt so stark vom Fahrplan ab wie VSC 72")
    print("  (exakte 2:1-Teilung). Bei ausgeglichenem Fahrplan bleibt U_dc nahe 33 kV")
    print("  (Rest: die Verluste). Und: Bei -100 kW sackt U_dc auf 32.2 kV - die")
    print("  Übermodulations-Warnung aus Szenario B springt an. Droop-Auslegung und")
    print("  DC-Spannungslage gehören zusammen.")
    return p2_sets, result


def plot(results):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib nicht installiert - Abbildung übersprungen;"
              " `uv run --with matplotlib ...` erzeugt sie)")
        return
    (p_sets, res_a), (u_sets, res_b), (p2_sets, res_c) = results
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    ax = axes[0]
    transfer = -res_a["vsc"]["p_to"][:, 0] / 1e3
    ax.plot(p_sets / 1e3, res_a["source"]["p"][:, 0] / 1e3, "o-", label="Slack A [kW]")
    ax.plot(p_sets / 1e3, res_a["source"]["p"][:, 1] / 1e3, "s-", label="Slack B [kW]")
    ax.plot(p_sets / 1e3, transfer, "^--", label="DC-Übertragung B→A [kW]")
    ax.set_xlabel("p_set VSC 72 [kW]")
    ax.set_title("A: p-Setpoint diktiert die Übertragung")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    m = res_b["node"]["u"][:, [2, 3]] / (K_MOD * res_b["dc_node"]["u"])
    ax.plot(u_sets / 1e3, m[:, 0], "o-", label="M VSC 71")
    ax.plot(u_sets / 1e3, m[:, 1], "s-", label="M VSC 72")
    ax.axhline(1.0, color="red", ls=":", label="M = 1 (Übermodulation)")
    ax.set_xlabel("u_dc_set [kV]")
    ax.set_title("B: Modulationsindex vs. DC-Spannungslage")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(p2_sets / 1e3, res_c["vsc"]["p_from"][:, 0] / 1e3, "o-", label="P_t VSC 71 (Gain 100)")
    ax.plot(p2_sets / 1e3, res_c["vsc"]["p_from"][:, 1] / 1e3, "s-", label="P_t VSC 72 (Gain 50)")
    ax2 = ax.twinx()
    ax2.plot(p2_sets / 1e3, res_c["dc_node"]["u"][:, 0] / 1e3, "^--", color="gray", label="U_dc [kV]")
    ax2.set_ylabel("U_dc [kV]", color="gray")
    ax.set_xlabel("droop_p_set VSC 72 [kW]")
    ax.set_title("C: Droop teilt Fahrplan-Abweichung 2:1")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig("hybrid_dc_link.png", dpi=140)
    print("\nAbbildung geschrieben: hybrid_dc_link.png")


def main():
    print(__doc__)
    results = (scenario_a(), scenario_b(), scenario_c())
    plot(results)


if __name__ == "__main__":
    main()
