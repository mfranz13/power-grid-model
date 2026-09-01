// SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
//
// SPDX-License-Identifier: MPL-2.0

#pragma once

#include "branch.hpp"

#include "../auxiliary/input.hpp"
#include "../auxiliary/update.hpp"
#include "../calculation_parameters.hpp"
#include "../common/common.hpp"
#include "../common/exception.hpp"

namespace power_grid_model {

// DC cable/line between two dc_nodes for the hybrid AC/DC power flow.
// Enters the Y-bus as a purely real series admittance 1/r1; r1 is the total loop resistance.
// The current base has no sqrt3 factor: i = p / u_dc.
class DcLine final : public Branch {
  public:
    using InputType = DcLineInput;
    using UpdateType = BranchUpdate;
    static constexpr char const* name = "dc_line";

    explicit DcLine(DcLineInput const& dc_line_input, double u1, double u2)
        : Branch{dc_line_input}, i_n_{dc_line_input.i_n}, base_i_{base_power_3p / u1} {
        if (cabs(u1 - u2) > numerical_tolerance) {
            throw ConflictVoltage{id(), from_node(), to_node(), u1, u2};
        }
        double const base_y = base_power_3p / (u1 * u1);
        y_series_ = 1.0 / (dc_line_input.r1 * base_y);
    }

    // override getter
    double base_i_from() const override { return base_i_; }
    double base_i_to() const override { return base_i_; }
    double loading(double /* max_s */, double max_i) const override { return is_nan(i_n_) ? 0.0 : max_i / i_n_; };
    double phase_shift() const override { return 0.0; }

  private:
    double i_n_;
    double base_i_;
    DoubleComplex y_series_; // real-valued

    BranchCalcParam<symmetric_t> sym_calc_param() const override { return calc_param_y_sym(y_series_, 0.0, 1.0); }
    BranchCalcParam<asymmetric_t> asym_calc_param() const override {
        // never reached: asymmetric calculation with DC components is rejected up front
        return calc_param_y_asym(y_series_, 0.0, y_series_, 0.0, 1.0);
    }
};

} // namespace power_grid_model
