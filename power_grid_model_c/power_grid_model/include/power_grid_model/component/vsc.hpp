// SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
//
// SPDX-License-Identifier: MPL-2.0

#pragma once

#include "branch.hpp"

#include "../auxiliary/input.hpp"
#include "../auxiliary/update.hpp"
#include "../calculation_parameters.hpp"
#include "../common/common.hpp"
#include "../common/enum.hpp"

namespace power_grid_model {

// AC/DC voltage source converter for the hybrid power flow.
// Convention: from-side = AC node, to-side = dc_node. p_set/q_set are in from-side branch
// convention (power flowing AC node -> converter). Contributes zero admittance to the Y-bus;
// the branch only reserves the (ac, dc) sparsity slots and joins AC and DC into one math model.
class Vsc final : public Branch {
  public:
    using InputType = VscInput;
    using UpdateType = VscUpdate;
    static constexpr char const* name = "vsc";

    explicit Vsc(VscInput const& vsc_input, double u1, double u2)
        : Branch{vsc_input},
          control_p_{vsc_input.control_p},
          p_set_{vsc_input.p_set},
          q_set_{vsc_input.q_set},
          u_dc_set_{vsc_input.u_dc_set},
          droop_p_set_{vsc_input.droop_p_set},
          droop_gain_{vsc_input.droop_gain},
          droop_u_ref_{vsc_input.droop_u_ref},
          loss_a_{vsc_input.loss_a},
          loss_b_{vsc_input.loss_b},
          loss_c_{vsc_input.loss_c},
          sn_{vsc_input.sn},
          control_q_{is_nan(vsc_input.control_q) ? VscQControlMode::q : vsc_input.control_q},
          u_ac_set_{vsc_input.u_ac_set},
          q_min_{vsc_input.q_min},
          q_max_{vsc_input.q_max},
          u_ac_rated_{u1},
          u_dc_rated_{u2},
          base_i_from_{base_power_3p / u1 / sqrt3},
          base_i_to_{base_power_3p / u2} {}

    // override getter
    double base_i_from() const override { return base_i_from_; }
    double base_i_to() const override { return base_i_to_; }
    double loading(double max_s, double /* max_i */) const override { return is_nan(sn_) ? 0.0 : max_s / sn_; };
    double phase_shift() const override { return 0.0; }

    VscCalcParam calc_param() const {
        return VscCalcParam{.status = static_cast<IntS>(edge_status()),
                            .control_p = control_p_,
                            .p_set = p_set_pu(),
                            .q_set = q_set_pu(),
                            .u_dc_set = u_dc_set_pu(),
                            .droop_p_set = droop_p_set_pu(),
                            .droop_gain = droop_gain_pu(),
                            .droop_u_ref = droop_u_ref_pu(),
                            .loss_a = loss_a_pu(),
                            .loss_b = loss_b_pu(),
                            .loss_c = loss_c_pu(),
                            .control_q = control_q_,
                            .u_ac_set = u_ac_set_pu(),
                            .q_min = q_min_pu(),
                            .q_max = q_max_pu()};
    }

    // control parameters in per unit, for the hybrid solver input
    constexpr VscControlMode control_p() const { return control_p_; }
    constexpr double p_set_pu() const { return p_set_ / base_power_3p; }
    constexpr double q_set_pu() const { return q_set_ / base_power_3p; }
    constexpr double u_dc_set_pu() const { return u_dc_set_ / u_dc_rated_; }
    constexpr double droop_p_set_pu() const { return droop_p_set_ / base_power_3p; }
    constexpr double droop_gain_pu() const { return droop_gain_ * u_dc_rated_ / base_power_3p; }
    double droop_u_ref_pu() const { return is_nan(droop_u_ref_) ? 1.0 : droop_u_ref_ / u_dc_rated_; }
    constexpr double loss_a_pu() const { return loss_a_ / base_power_3p; }
    constexpr double loss_b_pu() const { return loss_b_ * base_i_from_ / base_power_3p; }
    constexpr double loss_c_pu() const { return loss_c_ * base_i_from_ * base_i_from_ / base_power_3p; }
    constexpr VscQControlMode control_q() const { return control_q_; }
    constexpr double u_ac_set_pu() const { return u_ac_set_ / u_ac_rated_; }
    constexpr double q_min_pu() const { return q_min_ / base_power_3p; }
    constexpr double q_max_pu() const { return q_max_ / base_power_3p; }

    UpdateChange update(VscUpdate const& update_data) {
        assert(update_data.id == this->id() || is_nan(update_data.id));
        bool const changed = set_status(update_data.from_status, update_data.to_status);
        if (!is_nan(update_data.control_p)) {
            control_p_ = update_data.control_p;
        }
        update_member(p_set_, update_data.p_set);
        update_member(q_set_, update_data.q_set);
        update_member(u_dc_set_, update_data.u_dc_set);
        update_member(droop_p_set_, update_data.droop_p_set);
        update_member(droop_gain_, update_data.droop_gain);
        update_member(droop_u_ref_, update_data.droop_u_ref);
        update_member(loss_a_, update_data.loss_a);
        update_member(loss_b_, update_data.loss_b);
        update_member(loss_c_, update_data.loss_c);
        if (!is_nan(update_data.control_q)) {
            control_q_ = update_data.control_q;
        }
        update_member(u_ac_set_, update_data.u_ac_set);
        update_member(q_min_, update_data.q_min);
        update_member(q_max_, update_data.q_max);
        // setpoint/control changes reach the solver via PowerFlowInput, prepared per calculation
        return {.topo = changed, .param = changed};
    }

    VscUpdate inverse(VscUpdate update_data) const {
        assert(update_data.id == this->id() || is_nan(update_data.id));
        update_data = Branch::inverse(update_data);
        set_if_not_nan(update_data.control_p, control_p_);
        set_if_not_nan(update_data.p_set, p_set_);
        set_if_not_nan(update_data.q_set, q_set_);
        set_if_not_nan(update_data.u_dc_set, u_dc_set_);
        set_if_not_nan(update_data.droop_p_set, droop_p_set_);
        set_if_not_nan(update_data.droop_gain, droop_gain_);
        set_if_not_nan(update_data.droop_u_ref, droop_u_ref_);
        set_if_not_nan(update_data.loss_a, loss_a_);
        set_if_not_nan(update_data.loss_b, loss_b_);
        set_if_not_nan(update_data.loss_c, loss_c_);
        set_if_not_nan(update_data.control_q, control_q_);
        set_if_not_nan(update_data.u_ac_set, u_ac_set_);
        set_if_not_nan(update_data.q_min, q_min_);
        set_if_not_nan(update_data.q_max, q_max_);
        return update_data;
    }

  private:
    VscControlMode control_p_;
    // all setpoints and parameters are stored in SI units; getters convert to per unit
    double p_set_;
    double q_set_;
    double u_dc_set_;
    double droop_p_set_;
    double droop_gain_;
    double droop_u_ref_;
    double loss_a_;
    double loss_b_;
    double loss_c_;
    double sn_;
    VscQControlMode control_q_;
    double u_ac_set_;
    double q_min_;
    double q_max_;
    double u_ac_rated_;
    double u_dc_rated_;
    double base_i_from_;
    double base_i_to_;

    static void update_member(double& member, double update_value) {
        if (!is_nan(update_value)) {
            member = update_value;
        }
    }

    BranchCalcParam<symmetric_t> sym_calc_param() const override { return BranchCalcParam<symmetric_t>{}; }
    BranchCalcParam<asymmetric_t> asym_calc_param() const override { return BranchCalcParam<asymmetric_t>{}; }
};

} // namespace power_grid_model
