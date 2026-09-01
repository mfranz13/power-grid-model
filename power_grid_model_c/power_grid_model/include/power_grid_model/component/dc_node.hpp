// SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
//
// SPDX-License-Identifier: MPL-2.0

#pragma once

#include "node.hpp"

#include "../auxiliary/input.hpp"
#include "../common/common.hpp"

namespace power_grid_model {

// DC bus for the hybrid AC/DC power flow. u_rated is the rated DC voltage (pole-to-pole).
// Reuses NodeOutput: u_angle is always 0, q is always 0.
class DcNode final : public Node {
  public:
    using InputType = DcNodeInput;
    static constexpr char const* name = "dc_node";

    explicit DcNode(DcNodeInput const& dc_node_input) : Node{dc_node_input} {}
};

} // namespace power_grid_model
