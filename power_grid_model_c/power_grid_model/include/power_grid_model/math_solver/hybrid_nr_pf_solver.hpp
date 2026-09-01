// SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
//
// SPDX-License-Identifier: MPL-2.0

#pragma once

/*
Fork extension: unified hybrid AC/DC Newton-Raphson power flow (Zulkifli generalized formulation).

The AC part is identical to NewtonRaphsonPFSolver. DC buses and VSC converters are folded into the
same uniform 2x2-block bus system by exact symbolic elimination of the converter unknowns:

- DC buses: theta == 0 is enforced by replacing the Q-equation row with the identity
  d_theta = 0 (same mechanism as the PV-constraint row in the NR solver). With a purely real
  admittance and real voltages, the generic H/N/M/L formulas automatically produce the correct
  DC power-balance derivatives (H = M = L = 0, N = V_i * V_j * G).

- Converter DC-side power P_f is eliminated through the IEC 62751-2 loss equation:
      P_f = loss(P_t, Q_t, U_a) - P_t,   loss = a + b * I_t + c * I_t^2,   I_t = S / U_a,
  with S = sqrt(P_t^2 + Q_t^2). P_t and Q_t are the AC-side converter powers in from-side branch
  convention (power flowing AC bus -> converter); the converter injects -P_t - jQ_t at the AC bus
  and -P_f at the DC bus.

- u_dc control: the DC voltage is pinned, which frees the V column of that DC bus; the converter's
  P_t takes that column slot as an absolute increment (no V-scaling).

All converter Jacobian contributions land on the (a,a)/(a,d)/(d,a)/(d,d) block slots, which exist
in the sparsity pattern because every VSC is registered as a (zero-admittance) branch.

The derivative of the b-term of the loss polynomial is singular at S = 0; it is skipped there
(the c-term is smooth everywhere). Sign conventions follow the derivation in
hybrid_loadflow/docs/analytische_jacobi_unified.md, adapted to PGM's J * dx = PQ_sp - PQ_cal
convention with V-scaled voltage columns.
*/

#include "block_matrix.hpp"
#include "newton_raphson_pf_solver.hpp"
#include "sparse_lu_solver.hpp"
#include "y_bus.hpp"

#include "../calculation_parameters.hpp"
#include "../common/common.hpp"
#include "../common/enum.hpp"
#include "../common/exception.hpp"
#include "../common/grouped_index_vector.hpp"
#include "../common/logging.hpp"
#include "../common/three_phase_tensor.hpp"
#include "../common/timer.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <format>
#include <functional>
#include <limits>
#include <numeric>
#include <vector>

namespace power_grid_model::math_solver {

namespace hybrid_nr_pf {

using newton_raphson_pf::ComplexPower;
using newton_raphson_pf::PFJacBlock;
using newton_raphson_pf::PolarPhasor;

// below this per-unit converter apparent power the singular b-term derivative of the loss
// polynomial is skipped (same guard as the b_idx handling in acdcpf.generalized.system)
constexpr double s_epsilon = 1e-10;

template <symmetry_tag sym_type> class HybridNRPFSolver {
  public:
    using sym = sym_type;

    using SparseSolverType = SparseLUSolver<PFJacBlock<sym>, ComplexPower<sym>, PolarPhasor<sym>>;
    using BlockPermArray = typename SparseSolverType::BlockPermArray;

    HybridNRPFSolver(YBus<sym> const& y_bus, MathModelTopology const& topo)
        : n_bus_{y_bus.size()},
          data_jac_(y_bus.nnz_lu()),
          x_(y_bus.size()),
          del_x_pq_(y_bus.size()),
          sparse_solver_{y_bus.row_indptr_lu(), y_bus.col_indices_lu(), y_bus.lu_diag()},
          perm_(y_bus.size()),
          load_gens_per_bus_{std::cref(topo.load_gens_per_bus)},
          sources_per_bus_{std::cref(topo.sources_per_bus)},
          load_gen_type_{std::cref(topo.load_gen_type)},
          branch_bus_idx_{std::cref(topo.branch_bus_idx)} {}

    SolverOutput<sym> run_power_flow(YBus<sym> const& y_bus, PowerFlowInput<sym> const& input, double err_tol,
                                     Idx max_iter, bool cache_run, Logger& log) {
        if constexpr (!is_symmetric_v<sym>) {
            throw NotImplementedError{};
        } else {
            // keep copy, as reference might break batching (same pattern as IterativePFSolver)
            auto solver = *this;
            return solver.run_power_flow_impl(y_bus, input, err_tol, max_iter, cache_run, log);
        }
    }

  private:
    struct VscState {
        Idx branch{};    // math branch index
        Idx a{};         // AC math bus
        Idx d{};         // DC math bus
        Idx lu_da{};     // LU entry (d, a)
        Idx lu_ad{};     // LU entry (a, d), only used for u_dc control
        VscCalcParam param{};
        double p_t{}; // AC-side active power, from-side convention; solver state for u_dc control
        double p_f{}; // DC-side power, evaluated from the loss equation
    };

    Idx n_bus_;
    std::vector<PFJacBlock<sym>> data_jac_;
    std::vector<PolarPhasor<sym>> x_;
    std::vector<PolarPhasor<sym>> del_x_pq_;
    SparseSolverType sparse_solver_;
    BlockPermArray perm_;
    std::reference_wrapper<SparseGroupedIdxVector const> load_gens_per_bus_;
    std::reference_wrapper<DenseGroupedIdxVector const> sources_per_bus_;
    std::reference_wrapper<std::vector<LoadGenType> const> load_gen_type_;
    std::reference_wrapper<std::vector<BranchIdx> const> branch_bus_idx_;

    // per-run state
    std::vector<VscState> vsc_;
    IntSVector is_dc_bus_;
    IdxVector pinning_vsc_of_bus_; // per bus: index into vsc_ of the u_dc converter pinning it, or -1

    SolverOutput<sym> run_power_flow_impl(YBus<sym> const& y_bus, PowerFlowInput<sym> const& input, double err_tol,
                                          Idx max_iter, bool /*cache_run*/, Logger& log) {
        SolverOutput<sym> output;
        output.u.resize(n_bus_);
        double max_dev = std::numeric_limits<double>::infinity();

        Timer main_timer{log, LogEvent::math_solver};

        {
            Timer const sub_timer{log, LogEvent::initialize_calculation};
            initialize(y_bus, input, output);
        }

        Idx num_iter = 0;
        while (max_dev > err_tol || num_iter == 0) {
            if (num_iter++ == max_iter) {
                throw IterationDiverge{max_iter, max_dev, err_tol};
            }
            {
                Timer const sub_timer{log, LogEvent::prepare_matrices};
                prepare_matrix_and_rhs(y_bus, input, output.u);
            }
            {
                Timer const sub_timer{log, LogEvent::solve_sparse_linear_equation};
                sparse_solver_.prefactorize_and_solve(data_jac_, perm_, del_x_pq_, del_x_pq_);
            }
            {
                Timer const sub_timer{log, LogEvent::iterate_unknown};
                max_dev = iterate_unknown(output.u);
            }
        }

        {
            Timer const sub_timer{log, LogEvent::calculate_math_result};
            output.bus.resize(n_bus_); // no Q-limit machinery in the hybrid solver; default state
            detail::calculate_pf_result(y_bus, input, sources_per_bus_.get(), load_gens_per_bus_.get(), output,
                                        [this](Idx i) { return (load_gen_type_.get())[i]; });
            finalize_converter_output(output);
        }
        main_timer.stop();

        log.log(LogEvent::iterative_pf_solver_max_num_iter, num_iter);
        return output;
    }

    // ------------------- initialization -------------------

    void initialize(YBus<sym> const& y_bus, PowerFlowInput<sym> const& input, SolverOutput<sym>& output) {
        set_up_dc_state(y_bus, input);
        validate_dc_control(y_bus);
        set_converter_start_values(y_bus);
        linear_initial_guess(y_bus, input, output);
    }

    void set_up_dc_state(YBus<sym> const& y_bus, PowerFlowInput<sym> const& input) {
        is_dc_bus_.assign(n_bus_, 0);
        if (!input.is_dc_bus.empty()) {
            assert(std::ssize(input.is_dc_bus) == n_bus_);
            is_dc_bus_ = input.is_dc_bus;
        }
        pinning_vsc_of_bus_.assign(n_bus_, -1);
        vsc_.clear();
        auto const& branch_bus_idx = branch_bus_idx_.get();
        for (Idx k = 0; k != std::ssize(input.vsc); ++k) {
            Idx const branch = input.vsc_math_branch[k];
            auto const [a, d] = branch_bus_idx[branch];
            if (input.vsc[k].status == 0 || a < 0 || d < 0) {
                continue;
            }
            VscState state{.branch = branch,
                           .a = a,
                           .d = d,
                           .lu_da = lu_entry(y_bus, d, a),
                           .lu_ad = lu_entry(y_bus, a, d),
                           .param = input.vsc[k]};
            if (state.param.control_p == VscControlMode::u_dc) {
                if (pinning_vsc_of_bus_[d] != -1) {
                    throw DcControlConfigurationError{
                        std::format("more than one u_dc-controlling converter at DC bus {}", d)};
                }
                pinning_vsc_of_bus_[d] = std::ssize(vsc_);
            }
            vsc_.push_back(state);
        }
    }

    static Idx lu_entry(YBus<sym> const& y_bus, Idx row, Idx col) {
        IdxVector const& indptr = y_bus.row_indptr_lu();
        IdxVector const& indices = y_bus.col_indices_lu();
        for (Idx k = indptr[row]; k != indptr[row + 1]; ++k) {
            if (indices[k] == col) {
                return k;
            }
        }
        throw SparseMatrixError{}; // the VSC branch guarantees this entry; missing means internal error
    }

    // union-find over DC buses through conducting DC branches: every galvanically connected DC
    // subnetwork needs at least one voltage-anchoring converter, otherwise its conductance block
    // is singular (the eliminated-form analogue of the squareness check in acdcpf.generalized)
    void validate_dc_control(YBus<sym> const& y_bus) {
        if (std::ranges::none_of(is_dc_bus_, [](IntS f) { return f != 0; })) {
            if (!vsc_.empty()) {
                throw DcControlConfigurationError{"VSC converters present but no DC buses"};
            }
            return;
        }
        IdxVector parent(n_bus_);
        std::iota(parent.begin(), parent.end(), Idx{0});
        auto find = [&parent](Idx i) {
            while (parent[i] != i) {
                parent[i] = parent[parent[i]];
                i = parent[i];
            }
            return i;
        };
        auto const& branch_bus_idx = branch_bus_idx_.get();
        auto const& branch_param = y_bus.math_model_param().branch_param;
        for (Idx b = 0; b != std::ssize(branch_bus_idx); ++b) {
            auto const [i, j] = branch_bus_idx[b];
            if (i < 0 || j < 0 || is_dc_bus_[i] == 0 || is_dc_bus_[j] == 0) {
                continue;
            }
            if (cabs(branch_param[b].yft()) < numerical_tolerance) {
                continue; // disconnected or zero-admittance branch does not join the subnets
            }
            parent[find(i)] = find(j);
        }
        std::vector<char> has_anchor(n_bus_, 0);
        for (auto const& vsc : vsc_) {
            if (vsc.param.control_p == VscControlMode::u_dc || vsc.param.control_p == VscControlMode::droop) {
                has_anchor[find(vsc.d)] = 1;
            }
        }
        for (Idx i = 0; i != n_bus_; ++i) {
            if (is_dc_bus_[i] != 0 && has_anchor[find(i)] == 0) {
                throw DcControlConfigurationError{std::format(
                    "DC subnetwork of math bus {} has no u_dc- or droop-controlling converter", i)};
            }
        }
    }

    void set_converter_start_values(YBus<sym> const& y_bus) {
        // loss-free power balance estimate for the free converter powers per DC subnetwork,
        // mirroring GeneralizedSystem.initial_state(): sum(P_t) == 0 for a lossless DC net
        IdxVector parent(n_bus_);
        std::iota(parent.begin(), parent.end(), Idx{0});
        auto find = [&parent](Idx i) {
            while (parent[i] != i) {
                parent[i] = parent[parent[i]];
                i = parent[i];
            }
            return i;
        };
        auto const& branch_bus_idx = branch_bus_idx_.get();
        auto const& branch_param = y_bus.math_model_param().branch_param;
        for (Idx b = 0; b != std::ssize(branch_bus_idx); ++b) {
            auto const [i, j] = branch_bus_idx[b];
            if (i >= 0 && j >= 0 && is_dc_bus_[i] != 0 && is_dc_bus_[j] != 0 &&
                cabs(branch_param[b].yft()) >= numerical_tolerance) {
                parent[find(i)] = find(j);
            }
        }
        std::vector<double> fixed_p_t_sum(n_bus_, 0.0);
        IdxVector n_free(n_bus_, 0);
        auto const droop_start_p_t = [](VscCalcParam const& param) {
            return param.droop_p_set + param.droop_gain * (param.droop_u_ref - 1.0); // flat start U_dc = 1
        };
        for (auto const& vsc : vsc_) {
            switch (vsc.param.control_p) {
            case VscControlMode::p:
                fixed_p_t_sum[find(vsc.d)] += vsc.param.p_set;
                break;
            case VscControlMode::droop:
                fixed_p_t_sum[find(vsc.d)] += droop_start_p_t(vsc.param);
                break;
            default:
                ++n_free[find(vsc.d)];
                break;
            }
        }
        for (auto& vsc : vsc_) {
            switch (vsc.param.control_p) {
            case VscControlMode::p:
                vsc.p_t = vsc.param.p_set;
                break;
            case VscControlMode::droop:
                vsc.p_t = droop_start_p_t(vsc.param);
                break;
            default: {
                Idx const root = find(vsc.d);
                double const estimate = -fixed_p_t_sum[root] / static_cast<double>(std::max(n_free[root], Idx{1}));
                // keep the start away from the singular sqrt point of the b loss term
                vsc.p_t = std::abs(estimate) > 1e-3 ? estimate : 1e-3;
                break;
            }
            }
            vsc.p_f = converter_loss(vsc, 1.0) - vsc.p_t;
        }
    }

    // linear complex initial guess as in the NR solver; DC rows are replaced by an identity with
    // the start voltage as right-hand side, because the DC conductance block has no sources and
    // would otherwise be singular
    void linear_initial_guess(YBus<sym> const& y_bus, PowerFlowInput<sym> const& input, SolverOutput<sym>& output) {
        std::ranges::fill(data_jac_, PFJacBlock<sym>{});
        std::ranges::fill(del_x_pq_, PolarPhasor<sym>{});

        IdxVector const& map_lu_y_bus = y_bus.map_lu_y_bus();
        ComplexTensorVector<sym> const& ydata = y_bus.admittance();
        for (Idx jac_idx = 0; jac_idx < std::ssize(data_jac_); ++jac_idx) {
            Idx const k_y_bus = map_lu_y_bus[jac_idx];
            if (k_y_bus != -1) {
                set_linear_block(data_jac_[jac_idx], ydata[k_y_bus]);
            }
        }

        IdxVector const& bus_entry = y_bus.lu_diag();
        auto const& load_gens_per_bus = load_gens_per_bus_.get();
        auto const& sources_per_bus = sources_per_bus_.get();
        for (auto const& [bus_number, load_gens, sources] :
             enumerated_zip_sequence(load_gens_per_bus, sources_per_bus)) {
            Idx const diagonal_position = bus_entry[bus_number];
            add_linear_initial_guess_loads(load_gens, data_jac_[diagonal_position], input);
            add_linear_initial_guess_sources(sources, data_jac_[diagonal_position], del_x_pq_[bus_number], y_bus,
                                             input);
        }

        // overwrite DC rows with an identity and the DC start voltage
        IdxVector const& indptr = y_bus.row_indptr_lu();
        IdxVector const& indices = y_bus.col_indices_lu();
        for (Idx row = 0; row != n_bus_; ++row) {
            if (is_dc_bus_[row] == 0) {
                continue;
            }
            for (Idx k = indptr[row]; k != indptr[row + 1]; ++k) {
                data_jac_[k] = PFJacBlock<sym>{};
                if (indices[k] == row) {
                    add_diag(data_jac_[k].real_real(), RealValue<sym>{1.0});
                    add_diag(data_jac_[k].imag_imag(), RealValue<sym>{1.0});
                }
            }
            double u_start = 1.0;
            if (Idx const pin = pinning_vsc_of_bus_[row]; pin != -1) {
                u_start = vsc_[pin].param.u_dc_set;
            }
            del_x_pq_[row].i_real() = RealValue<sym>{u_start};
            del_x_pq_[row].i_imag() = RealValue<sym>{0.0};
        }

        sparse_solver_.prefactorize_and_solve(data_jac_, perm_, del_x_pq_, del_x_pq_);

        for (Idx i = 0; i != n_bus_; ++i) {
            output.u[i] =
                ComplexValue<sym>{RealValue<sym>{del_x_pq_[i].u_real()}, RealValue<sym>{del_x_pq_[i].u_imag()}};
            x_[i].v() = cabs(output.u[i]);
            x_[i].theta() = arg(output.u[i]);
        }
    }

    // ------------------- per-iteration assembly -------------------

    void prepare_matrix_and_rhs(YBus<sym> const& y_bus, PowerFlowInput<sym> const& input,
                                ComplexValueVector<sym> const& u) {
        IdxVector const& bus_entry = y_bus.lu_diag();
        prepare_matrix_and_rhs_from_network_perspective(y_bus, u, bus_entry);

        auto const& load_gens_per_bus = load_gens_per_bus_.get();
        auto const& sources_per_bus = sources_per_bus_.get();
        std::vector<LoadGenType> const& load_gen_type = load_gen_type_.get();
        for (auto const& [bus_number, load_gens, sources] :
             enumerated_zip_sequence(load_gens_per_bus, sources_per_bus)) {
            Idx const diagonal_position = bus_entry[bus_number];
            add_loads(load_gens, bus_number, diagonal_position, input, load_gen_type);
            add_sources(sources, bus_number, diagonal_position, y_bus, input, u);
        }

        apply_dc_constraints(y_bus);
        add_converters(bus_entry);
    }

    // identical to NewtonRaphsonPFSolver::calculate_hnml (kept local to avoid touching upstream)
    static PFJacBlock<sym> calculate_hnml(ComplexTensor<sym> const& yij, ComplexValue<sym> const& ui,
                                          ComplexValue<sym> const& uj) {
        PFJacBlock<sym> block{};
        ComplexTensor<sym> const power_flow_ij = vector_outer_product(ui, conj(uj)) * conj(yij);
        block.h() = imag(power_flow_ij);
        block.n() = real(power_flow_ij);
        block.m() = -block.n();
        block.l() = block.h();
        return block;
    }

    void prepare_matrix_and_rhs_from_network_perspective(YBus<sym> const& y_bus, ComplexValueVector<sym> const& u,
                                                         IdxVector const& bus_entry) {
        IdxVector const& indptr = y_bus.row_indptr_lu();
        IdxVector const& indices = y_bus.col_indices_lu();
        IdxVector const& map_lu_y_bus = y_bus.map_lu_y_bus();
        ComplexTensorVector<sym> const& ydata = y_bus.admittance();

        for (Idx row = 0; row != n_bus_; ++row) {
            del_x_pq_[row].p() = RealValue<sym>{0.0};
            del_x_pq_[row].q() = RealValue<sym>{0.0};
            for (Idx k = indptr[row]; k != indptr[row + 1]; ++k) {
                Idx const k_y_bus = map_lu_y_bus[k];
                if (k_y_bus == -1) {
                    data_jac_[k] = PFJacBlock<sym>{};
                    continue;
                }
                Idx const j = indices[k];
                data_jac_[k] = calculate_hnml(ydata[k_y_bus], u[row], u[j]);
                del_x_pq_[row].p() -= sum_row(data_jac_[k].n());
                del_x_pq_[row].q() -= sum_row(data_jac_[k].h());
            }
            Idx const k = bus_entry[row];
            add_diag(data_jac_[k].h(), del_x_pq_[row].q());
            add_diag(data_jac_[k].n(), -del_x_pq_[row].p());
            add_diag(data_jac_[k].m(), -del_x_pq_[row].p());
            add_diag(data_jac_[k].l(), -del_x_pq_[row].q());
        }
    }

    // DC row/column structure, applied after the network/load/source assembly and before the
    // converter entries are written:
    // - the V column of every u_dc-pinned bus is cleared in all rows (the voltage is no longer an
    //   unknown; the slot is reused for the converter's delta-P_t, written in add_converters)
    // - the Q row of every DC bus is replaced by the identity d_theta = 0
    void apply_dc_constraints(YBus<sym> const& y_bus) {
        if (vsc_.empty() && std::ranges::none_of(is_dc_bus_, [](IntS f) { return f != 0; })) {
            return;
        }
        IdxVector const& indptr = y_bus.row_indptr_lu();
        IdxVector const& indices = y_bus.col_indices_lu();
        for (Idx row = 0; row != n_bus_; ++row) {
            bool const dc_row = is_dc_bus_[row] != 0;
            for (Idx k = indptr[row]; k != indptr[row + 1]; ++k) {
                Idx const col = indices[k];
                if (pinning_vsc_of_bus_[col] != -1) {
                    data_jac_[k].n() = RealTensor<sym>{0.0};
                    data_jac_[k].l() = RealTensor<sym>{0.0};
                }
                if (dc_row) {
                    // theta-pad row: replace the Q equation by d_theta = 0
                    data_jac_[k].m() = RealTensor<sym>{0.0};
                    data_jac_[k].l() = RealTensor<sym>{0.0};
                    if (row == col) {
                        add_diag(data_jac_[k].m(), RealValue<sym>{1.0});
                    }
                }
            }
            if (dc_row) {
                del_x_pq_[row].q() = RealValue<sym>{0.0};
            }
        }
    }

    void add_converters(IdxVector const& bus_entry) {
        if constexpr (is_symmetric_v<sym>) {
            for (auto& vsc : vsc_) {
                double const v_a = x_[vsc.a].v();
                double const v_d = x_[vsc.d].v();
                if (vsc.param.control_p == VscControlMode::droop) {
                    // eliminated droop equation: P_t is a linear function of U_dc
                    vsc.p_t = vsc.param.droop_p_set + vsc.param.droop_gain * (vsc.param.droop_u_ref - v_d);
                }
                double const p_t = vsc.p_t;
                double const q_t = vsc.param.q_set;
                double const s = std::hypot(p_t, q_t);
                double const i_t = s / v_a;
                double const loss = vsc.param.loss_a + vsc.param.loss_b * i_t + vsc.param.loss_c * i_t * i_t;
                vsc.p_f = loss - p_t;

                // d(loss)/dU_a * V_a and d(loss)/dP_t, with the b-term guarded at S = 0
                double d_loss_du_va = -2.0 * vsc.param.loss_c * i_t * i_t;
                double d_loss_dpt = 2.0 * vsc.param.loss_c * p_t / (v_a * v_a);
                if (vsc.param.loss_b != 0.0 && s > s_epsilon) {
                    d_loss_du_va -= vsc.param.loss_b * i_t;
                    d_loss_dpt += vsc.param.loss_b * p_t / (s * v_a);
                }

                // specified injections: p_sp_a -= P_t, q_sp_a -= Q_t, p_sp_d -= P_f
                del_x_pq_[vsc.a].p() -= p_t;
                del_x_pq_[vsc.a].q() -= q_t;
                del_x_pq_[vsc.d].p() -= vsc.p_f;

                // J.N(d,a) -= dP_sp_d/dV_a * V_a = -d(loss)/dU_a * V_a
                data_jac_[vsc.lu_da].n() += d_loss_du_va;

                if (vsc.param.control_p == VscControlMode::u_dc) {
                    // the V slot of bus d carries delta-P_t (absolute, unscaled)
                    // J.N(a,d) -= dP_sp_a/dP_t = -1
                    data_jac_[vsc.lu_ad].n() += 1.0;
                    // J.N(d,d) -= dP_sp_d/dP_t = -(d(loss)/dP_t - 1)
                    data_jac_[bus_entry[vsc.d]].n() += d_loss_dpt - 1.0;
                } else if (vsc.param.control_p == VscControlMode::droop && pinning_vsc_of_bus_[vsc.d] == -1) {
                    // dP_t/dU_dc = -gain; if the bus is pinned by a u_dc converter, U_dc is fixed
                    // and P_t is effectively constant, so no column entries are written there
                    double const dpt_dud = -vsc.param.droop_gain;
                    // J.N(a,d) -= dP_sp_a/dV_d * V_d = -dP_t/dU_dc * V_d
                    data_jac_[vsc.lu_ad].n() += dpt_dud * v_d;
                    // J.N(d,d) -= dP_sp_d/dV_d * V_d = +(d(loss)/dP_t - 1) * dP_t/dU_dc * V_d
                    data_jac_[bus_entry[vsc.d]].n() += (d_loss_dpt - 1.0) * dpt_dud * v_d;
                }
            }
        }
    }

    double iterate_unknown(ComplexValueVector<sym>& u) {
        double max_dev = 0.0;
        for (Idx i = 0; i != n_bus_; ++i) {
            if constexpr (is_symmetric_v<sym>) {
                if (Idx const pin = pinning_vsc_of_bus_[i]; pin != -1) {
                    // voltage is pinned; the V slot carries delta-P_t of the pinning converter
                    double const del_p_t = del_x_pq_[i].v();
                    vsc_[pin].p_t += del_p_t;
                    max_dev = std::max(max_dev, std::abs(del_p_t));
                    u[i] = DoubleComplex{x_[i].v(), 0.0};
                    continue;
                }
            }
            x_[i].theta() += del_x_pq_[i].theta();
            x_[i].v() += x_[i].v() * del_x_pq_[i].v();
            ComplexValue<sym> const u_tmp = x_[i].v() * exp(1.0i * x_[i].theta());
            double const dev = max_val(cabs(u_tmp - u[i]));
            max_dev = std::max(dev, max_dev);
            u[i] = u_tmp;
        }
        return max_dev;
    }

    // ------------------- result -------------------

    double converter_loss(VscState const& vsc, double v_a) const {
        double const i_t = std::hypot(vsc.p_t, vsc.param.q_set) / v_a;
        return vsc.param.loss_a + vsc.param.loss_b * i_t + vsc.param.loss_c * i_t * i_t;
    }

    void finalize_converter_output(SolverOutput<sym>& output) {
        if constexpr (is_symmetric_v<sym>) {
            for (auto& vsc : vsc_) {
                double const v_a = x_[vsc.a].v();
                double const v_d = x_[vsc.d].v();
                if (vsc.param.control_p == VscControlMode::droop) {
                    // refresh with the final DC voltage (the last iterate moved U_dc once more)
                    vsc.p_t = vsc.param.droop_p_set + vsc.param.droop_gain * (vsc.param.droop_u_ref - v_d);
                }
                double const p_f = converter_loss(vsc, v_a) - vsc.p_t;
                auto& branch = output.branch[vsc.branch];
                DoubleComplex const s_f{vsc.p_t, vsc.param.q_set};
                branch.s_f = s_f;
                branch.i_f = conj(s_f / output.u[vsc.a]);
                branch.s_t = DoubleComplex{p_f, 0.0};
                branch.i_t = DoubleComplex{p_f / v_d, 0.0};
            }
        }
    }

    // ------------------- copies of NR helper functions (loads/sources/linear guess) -------------------

    void add_loads(IdxRange const& load_gens, Idx bus_number, Idx diagonal_position, PowerFlowInput<sym> const& input,
                   std::vector<LoadGenType> const& load_gen_type) {
        using enum LoadGenType;
        for (Idx const load_number : load_gens) {
            switch (LoadGenType const type = load_gen_type[load_number]; type) {
            case const_pq:
                del_x_pq_[bus_number].p() += real(input.s_injection[load_number]);
                del_x_pq_[bus_number].q() += imag(input.s_injection[load_number]);
                break;
            case const_y:
                del_x_pq_[bus_number].p() +=
                    real(input.s_injection[load_number]) * x_[bus_number].v() * x_[bus_number].v();
                del_x_pq_[bus_number].q() +=
                    imag(input.s_injection[load_number]) * x_[bus_number].v() * x_[bus_number].v();
                add_diag(data_jac_[diagonal_position].n(),
                         -real(input.s_injection[load_number]) * 2.0 * x_[bus_number].v() * x_[bus_number].v());
                add_diag(data_jac_[diagonal_position].l(),
                         -imag(input.s_injection[load_number]) * 2.0 * x_[bus_number].v() * x_[bus_number].v());
                break;
            case const_i:
                del_x_pq_[bus_number].p() += real(input.s_injection[load_number]) * x_[bus_number].v();
                del_x_pq_[bus_number].q() += imag(input.s_injection[load_number]) * x_[bus_number].v();
                add_diag(data_jac_[diagonal_position].n(),
                         -real(input.s_injection[load_number]) * x_[bus_number].v());
                add_diag(data_jac_[diagonal_position].l(),
                         -imag(input.s_injection[load_number]) * x_[bus_number].v());
                break;
            default:
                throw MissingCaseForEnumError("Jacobian and deviation calculation", type);
            }
        }
    }

    void add_sources(IdxRange const& sources, Idx bus_number, Idx diagonal_position, YBus<sym> const& y_bus,
                     PowerFlowInput<sym> const& input, ComplexValueVector<sym> const& u) {
        for (Idx const source_number : sources) {
            ComplexTensor<sym> const y_ref = y_bus.math_model_param().source_param[source_number].template y_ref<sym>();
            ComplexValue<sym> const u_ref{input.source[source_number]};
            PFJacBlock<sym> block_mm = calculate_hnml(y_ref, u[bus_number], u[bus_number]);
            PFJacBlock<sym> block_ms = calculate_hnml(-y_ref, u[bus_number], u_ref);
            RealValue<sym> const p_cal = sum_row(block_mm.n() + block_ms.n());
            RealValue<sym> const q_cal = sum_row(block_mm.h() + block_ms.h());
            add_diag(block_mm.h(), -q_cal);
            add_diag(block_mm.n(), p_cal);
            add_diag(block_mm.m(), p_cal);
            add_diag(block_mm.l(), q_cal);
            del_x_pq_[bus_number].p() -= p_cal;
            del_x_pq_[bus_number].q() -= q_cal;
            data_jac_[diagonal_position].h() += block_mm.h();
            data_jac_[diagonal_position].n() += block_mm.n();
            data_jac_[diagonal_position].m() += block_mm.m();
            data_jac_[diagonal_position].l() += block_mm.l();
        }
    }

    void add_linear_initial_guess_loads(IdxRange const& load_gens, PFJacBlock<sym>& block,
                                        PowerFlowInput<sym> const& input) {
        for (Idx const load_number : load_gens) {
            ComplexValue<sym> const& s = input.s_injection[load_number];
            ComplexValue<sym> const y_load = -conj(s);
            add_diag(block.real_imag(), -imag(y_load));
            add_diag(block.real_real(), real(y_load));
            add_diag(block.imag_imag(), real(y_load));
            add_diag(block.imag_real(), imag(y_load));
        }
    }

    void add_linear_initial_guess_sources(IdxRange const& sources, PFJacBlock<sym>& block, PolarPhasor<sym>& rhs,
                                          YBus<sym> const& y_bus, PowerFlowInput<sym> const& input) {
        for (Idx const source_number : sources) {
            ComplexTensor<sym> const y_source =
                y_bus.math_model_param().source_param[source_number].template y_ref<sym>();
            ComplexValue<sym> const u_source{input.source[source_number]};
            block.real_imag() -= imag(y_source);
            block.real_real() += real(y_source);
            block.imag_imag() += real(y_source);
            block.imag_real() += imag(y_source);
            ComplexValue<sym> const i_rhs = dot(y_source, u_source);
            rhs.i_real() += real(i_rhs);
            rhs.i_imag() += imag(i_rhs);
        }
    }

    static void set_linear_block(PFJacBlock<sym>& block, ComplexTensor<sym> const& y) {
        RealTensor<sym> const g = real(y);
        RealTensor<sym> const b = imag(y);
        block.real_imag() = -b;
        block.real_real() = g;
        block.imag_imag() = g;
        block.imag_real() = b;
    }
};

} // namespace hybrid_nr_pf

using hybrid_nr_pf::HybridNRPFSolver;

} // namespace power_grid_model::math_solver
