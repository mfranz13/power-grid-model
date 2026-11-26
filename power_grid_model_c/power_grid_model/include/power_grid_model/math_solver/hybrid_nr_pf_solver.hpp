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

- u_ac control: the same swap on the AC side. The voltage magnitude of the AC bus is pinned and
  the converter's Q_t takes the freed V column slot as an absolute increment; theta and both the P
  and Q equations of that bus stay active. Not allowed at a bus with a source, which already fixes
  the voltage there. Reactive limits q_min/q_max (fork extension beyond the paper) switch the
  converter like a PV bus: Q_t is clamped and the voltage pin released, without switch-back.

All converter Jacobian contributions land on the (a,a)/(a,d)/(d,a)/(d,d) block slots, which exist
in the sparsity pattern because every VSC is registered as a (zero-admittance) branch.

Initialization (linear, flat, average_source) as in the NR solver; DC buses start at u_dc_set or 1 pu.

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
using newton_raphson_pf::do_limit_check;
using newton_raphson_pf::limit_check_at_iteration;
using newton_raphson_pf::no_limit_check;
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
          phase_shift_{std::cref(topo.phase_shift)},
          load_gens_per_bus_{std::cref(topo.load_gens_per_bus)},
          sources_per_bus_{std::cref(topo.sources_per_bus)},
          load_gen_type_{std::cref(topo.load_gen_type)},
          branch_bus_idx_{std::cref(topo.branch_bus_idx)},
          bus_control_(y_bus.size()),
          voltage_regulators_per_load_gen_{std::cref(topo.voltage_regulators_per_load_gen)},
          clamped_regulators_per_load_gen_(topo.load_gen_type.size(), RealValue<sym>{nan}) {}

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
        double q_t{}; // AC-side reactive power, from-side convention; solver state for u_ac control
        double p_f{}; // DC-side power, evaluated from the loss equation
    };

    // as in NewtonRaphsonPFSolver, without its partial-rebuild flag (the hybrid always rebuilds fully)
    struct QLimitState {
        bool has_q_limits{false};
        LimitViolation limit_violated{LimitViolation::none};
        double bus_q_min{0}; // start with 0, unspecified limit on regulator should change this to NaN
        double bus_q_max{0};
    };

    struct BusControlState {
        BusType type{BusType::pq};
        DoubleComplex u_ref{};
        QLimitState q_limit{};
    };

    Idx n_bus_;
    std::vector<PFJacBlock<sym>> data_jac_;
    std::vector<PolarPhasor<sym>> x_;
    std::vector<PolarPhasor<sym>> del_x_pq_;
    SparseSolverType sparse_solver_;
    BlockPermArray perm_;
    std::reference_wrapper<DoubleVector const> phase_shift_;
    std::reference_wrapper<SparseGroupedIdxVector const> load_gens_per_bus_;
    std::reference_wrapper<DenseGroupedIdxVector const> sources_per_bus_;
    std::reference_wrapper<std::vector<LoadGenType> const> load_gen_type_;
    std::reference_wrapper<std::vector<BranchIdx> const> branch_bus_idx_;
    std::vector<BusControlState> bus_control_;
    std::reference_wrapper<DenseGroupedIdxVector const> voltage_regulators_per_load_gen_;
    // store clamped Q-value for a load_gen in case of a limit violation to avoid recalculation in add_loads()
    std::vector<RealValue<sym>> clamped_regulators_per_load_gen_;
    // limit check control: -1 = no limit check, 0 = check limit, >0 = wait this many iterations before checking
    Idx limit_check_countdown_{no_limit_check};

    // per-run state
    std::vector<VscState> vsc_;
    IntSVector is_dc_bus_;
    IdxVector pinning_vsc_of_bus_; // per bus: index into vsc_ of the u_dc converter pinning it, or -1
    IdxVector pinning_q_vsc_of_bus_; // per bus: index into vsc_ of the u_ac converter pinning it, or -1

    SolverOutput<sym> run_power_flow_impl(YBus<sym> const& y_bus, PowerFlowInput<sym> const& input, double err_tol,
                                          Idx max_iter, bool cache_run, Logger& log) {
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
                max_dev = iterate_unknown(output.u, err_tol, cache_run);
            }
        }

        {
            Timer const sub_timer{log, LogEvent::calculate_math_result};
            finalize_result(output);
            detail::calculate_pf_result(y_bus, input, sources_per_bus_.get(), load_gens_per_bus_.get(), output,
                                        [this](Idx i) { return (load_gen_type_.get())[i]; },
                                        converter_bus_injection());
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

        std::ranges::fill(bus_control_, BusControlState{});
        std::ranges::fill(clamped_regulators_per_load_gen_, RealValue<sym>{nan});
        bool const has_usable_limits = set_bus_types_and_q_limits(input) || has_limited_q_converters();
        limit_check_countdown_ = has_usable_limits ? limit_check_at_iteration : no_limit_check;
        for (Idx i = 0; i != n_bus_; ++i) {
            if (bus_control_[i].type == BusType::pv && pinning_q_vsc_of_bus_[i] != -1) {
                throw DcControlConfigurationError{
                    std::format("voltage regulator and u_ac-controlling converter at the same AC bus {}", i)};
            }
        }

        set_converter_start_values(y_bus);
        switch (input.initialization) {
            using enum CalculationInitialization;
        case flat:
            make_flat_start(input, output.u);
            break;
        case average_source:
            make_average_source_start(input, output.u);
            break;
        default:
            linear_initial_guess(y_bus, input, output);
            break;
        }

        for (Idx i = 0; i != n_bus_; ++i) {
            if (is_dc_bus_[i] != 0) {
                Idx const pin = pinning_vsc_of_bus_[i];
                output.u[i] = ComplexValue<sym>{pin != -1 ? vsc_[pin].param.u_dc_set : 1.0};
            }
        }
        set_reference_voltage_for_pv_buses(output.u);
        for (Idx i = 0; i != n_bus_; ++i) {
            x_[i].v() = cabs(output.u[i]);
            x_[i].theta() = arg(output.u[i]);
        }
        // after the polar refresh so that the pinned magnitude is exact; PV and u_ac-pinned buses are disjoint
        if constexpr (is_symmetric_v<sym>) {
            for (Idx i = 0; i != n_bus_; ++i) {
                if (Idx const pin = pinning_q_vsc_of_bus_[i]; pin != -1) {
                    x_[i].v() = vsc_[pin].param.u_ac_set;
                    output.u[i] = x_[i].v() * exp(1.0i * x_[i].theta());
                }
            }
        }
    }

    // average reference voltage of all sources, with the topological phase shift of the source buses offset
    // (copy of IterativePFSolver::average_source_voltage)
    DoubleComplex average_source_voltage(PowerFlowInput<sym> const& input) const {
        std::vector<double> const& phase_shift = phase_shift_.get();
        DoubleComplex sum_u_ref = 0.0;
        for (auto const& [bus, sources] : enumerated_zip_sequence(sources_per_bus_.get())) {
            for (Idx const source : sources) {
                sum_u_ref += input.source[source] * std::exp(1.0i * -phase_shift[bus]);
            }
        }
        return sum_u_ref / static_cast<double>(input.source.size());
    }

    // copy of IterativePFSolver::make_average_source_start
    void make_average_source_start(PowerFlowInput<sym> const& input, ComplexValueVector<sym>& u) const {
        std::vector<double> const& phase_shift = phase_shift_.get();
        DoubleComplex const u_ref = average_source_voltage(input);
        for (Idx i = 0; i != n_bus_; ++i) {
            u[i] = ComplexValue<sym>{u_ref * std::exp(1.0i * phase_shift[i])};
        }
    }

    // copy of NewtonRaphsonPFSolver::make_flat_start
    void make_flat_start(PowerFlowInput<sym> const& input, ComplexValueVector<sym>& u) const {
        std::vector<double> const& phase_shift = phase_shift_.get();
        double const source_angle = arg(average_source_voltage(input));
        for (auto const& [bus, sources] : enumerated_zip_sequence(sources_per_bus_.get())) {
            if (sources.empty()) {
                u[bus] = ComplexValue<sym>{std::exp(1.0i * (source_angle + phase_shift[bus]))};
                continue;
            }
            DoubleComplex u_ref{};
            for (Idx const source : sources) {
                u_ref += input.source[source];
            }
            u[bus] = ComplexValue<sym>{u_ref / static_cast<double>(std::ranges::size(sources))};
        }
    }

    auto set_bus_types_and_q_limits(PowerFlowInput<sym> const& input) {
        auto const& voltage_regulators_per_load_gen = voltage_regulators_per_load_gen_.get();

        auto has_usable_q_limits = false;
        for (auto const& [bus_idx, load_gens, sources] :
             enumerated_zip_sequence(load_gens_per_bus_.get(), sources_per_bus_.get())) {
            auto& bus_control = bus_control_[bus_idx];
            if (!sources.empty()) {
                bus_control.type = BusType::slack;
                // don't need to calculate bus limits for slack buses, as active sources and regulators on the same node
                // are disallowed
                continue;
            }

            for (Idx const load_gen_idx : load_gens) {
                if (input.load_gen_status.empty() || input.load_gen_status[load_gen_idx] == status_off) {
                    continue;
                }
                for (Idx const voltage_regulator_idx :
                     voltage_regulators_per_load_gen.get_element_range(load_gen_idx)) {
                    auto const& regulator = input.voltage_regulator[voltage_regulator_idx];
                    if (regulator.status != status_off) {
                        bus_control.type = BusType::pv;
                        bus_control.u_ref = regulator.u_ref; // all regulators on the bus must have the same u_ref

                        // unlimited load_gen/regulator on a bus makes the whole bus unlimited,
                        // because `bus_limit = limit_reg1 + NaN = NaN`
                        bus_control.q_limit.bus_q_min += regulator.q_min;
                        bus_control.q_limit.bus_q_max += regulator.q_max;
                    }
                }
            }

            if (bus_control.type == BusType::pv) {
                bus_control.q_limit.has_q_limits =
                    !is_nan(bus_control.q_limit.bus_q_min) || !is_nan(bus_control.q_limit.bus_q_max);
                if (bus_control.q_limit.has_q_limits) {
                    has_usable_q_limits = true;
                }
            }
        }
        return has_usable_q_limits;
    }

    void set_reference_voltage_for_pv_buses(ComplexValueVector<sym>& u) {
        for (Idx i = 0; i != n_bus_; ++i) {
            if (bus_control_[i].type == BusType::pv) {
                u[i] = bus_control_[i].u_ref * phase_shift(u[i]);
            }
        }
    }

    void set_up_dc_state(YBus<sym> const& y_bus, PowerFlowInput<sym> const& input) {
        is_dc_bus_.assign(n_bus_, 0);
        if (!input.is_dc_bus.empty()) {
            assert(std::ssize(input.is_dc_bus) == n_bus_);
            is_dc_bus_ = input.is_dc_bus;
        }
        pinning_vsc_of_bus_.assign(n_bus_, -1);
        pinning_q_vsc_of_bus_.assign(n_bus_, -1);
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
            if (state.param.control_q == VscQControlMode::u_ac) {
                if (pinning_q_vsc_of_bus_[a] != -1) {
                    throw DcControlConfigurationError{
                        std::format("more than one u_ac-controlling converter at AC bus {}", a)};
                }
                if (!std::ranges::empty(sources_per_bus_.get().get_element_range(a))) {
                    throw DcControlConfigurationError{
                        std::format("u_ac-controlling converter at AC bus {} with a source", a)};
                }
                pinning_q_vsc_of_bus_[a] = std::ssize(vsc_);
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
            vsc.q_t = vsc.param.control_q == VscQControlMode::u_ac ? 0.0 : vsc.param.q_set;
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
        }
    }

    // ------------------- per-iteration assembly -------------------

    // The Q-limit check reads del_x_pq_ and therefore runs after the converter injections are in;
    // the PV rows are then overwritten before the DC column structure and the converter Jacobian
    // entries are applied (a PV bus is never u_ac-pinned, so no converter entry lands in a PV Q row).
    void prepare_matrix_and_rhs(YBus<sym> const& y_bus, PowerFlowInput<sym> const& input,
                                ComplexValueVector<sym> const& u) {
        build_jacobian_and_rhs(y_bus, input, u);

        if (limit_check_countdown_ > do_limit_check) {
            --limit_check_countdown_;
        }
        bool buses_switched = false;
        if (limit_check_countdown_ == do_limit_check) {
            // no short-circuit: both kinds of limits are checked in the same iteration
            buses_switched = enforce_q_limits(input);
            buses_switched = enforce_converter_q_limits() || buses_switched;
        }
        if (buses_switched) {
            build_jacobian_and_rhs(y_bus, input, u);
        }
        apply_pv_constraints(y_bus);
        apply_dc_constraints(y_bus);
        add_converter_jacobian(y_bus.lu_diag());
    }

    void build_jacobian_and_rhs(YBus<sym> const& y_bus, PowerFlowInput<sym> const& input,
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

        add_converter_injections();
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
    // converter Jacobian entries are written:
    // - the V column of every u_dc- or u_ac-pinned bus is cleared in all rows (the voltage is no
    //   longer an unknown; the slot is reused for the converter's delta-P_t or delta-Q_t, written
    //   in add_converter_jacobian)
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
                if (pinning_vsc_of_bus_[col] != -1 || pinning_q_vsc_of_bus_[col] != -1) {
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

    void add_converter_injections() {
        if constexpr (is_symmetric_v<sym>) {
            for (auto& vsc : vsc_) {
                if (vsc.param.control_p == VscControlMode::droop) {
                    // eliminated droop equation: P_t is a linear function of U_dc
                    vsc.p_t = vsc.param.droop_p_set + vsc.param.droop_gain * (vsc.param.droop_u_ref - x_[vsc.d].v());
                }
                vsc.p_f = converter_loss(vsc, x_[vsc.a].v()) - vsc.p_t;

                // specified injections: p_sp_a -= P_t, q_sp_a -= Q_t, p_sp_d -= P_f
                del_x_pq_[vsc.a].p() -= vsc.p_t;
                del_x_pq_[vsc.a].q() -= vsc.q_t;
                del_x_pq_[vsc.d].p() -= vsc.p_f;
            }
        }
    }

    // reads P_t/Q_t as refreshed by add_converter_injections
    void add_converter_jacobian(IdxVector const& bus_entry) {
        if constexpr (is_symmetric_v<sym>) {
            for (auto const& vsc : vsc_) {
                double const v_a = x_[vsc.a].v();
                double const v_d = x_[vsc.d].v();
                double const p_t = vsc.p_t;
                double const q_t = vsc.q_t;
                double const s = std::hypot(p_t, q_t);
                double const i_t = s / v_a;

                // d(loss)/dU_a * V_a, d(loss)/dP_t and d(loss)/dQ_t, with the b-term guarded at S = 0
                double d_loss_du_va = -2.0 * vsc.param.loss_c * i_t * i_t;
                double d_loss_dpt = 2.0 * vsc.param.loss_c * p_t / (v_a * v_a);
                double d_loss_dqt = 2.0 * vsc.param.loss_c * q_t / (v_a * v_a);
                if (vsc.param.loss_b != 0.0 && s > s_epsilon) {
                    d_loss_du_va -= vsc.param.loss_b * i_t;
                    d_loss_dpt += vsc.param.loss_b * p_t / (s * v_a);
                    d_loss_dqt += vsc.param.loss_b * q_t / (s * v_a);
                }

                if (Idx const q_pin = pinning_q_vsc_of_bus_[vsc.a]; q_pin == -1) {
                    // J.N(d,a) -= dP_sp_d/dV_a * V_a = -d(loss)/dU_a * V_a
                    data_jac_[vsc.lu_da].n() += d_loss_du_va;
                } else if (&vsc_[q_pin] == &vsc) {
                    // the V slot of bus a carries delta-Q_t (absolute, unscaled)
                    // J.N(d,a) -= dP_sp_d/dQ_t = -d(loss)/dQ_t
                    data_jac_[vsc.lu_da].n() += d_loss_dqt;
                    // J.L(a,a) -= dQ_sp_a/dQ_t = -1
                    data_jac_[bus_entry[vsc.a]].l() += 1.0;
                }
                // else: U_a is pinned by another converter, so this converter has no entries in
                // that column (its powers do not depend on the other converter's Q_t)

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

    double iterate_unknown(ComplexValueVector<sym>& u, double err_tol, bool cache_run) {
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
                if (Idx const pin = pinning_q_vsc_of_bus_[i]; pin != -1) {
                    // voltage magnitude is pinned; the V slot carries delta-Q_t of the pinning converter
                    double const del_q_t = del_x_pq_[i].v();
                    vsc_[pin].q_t += del_q_t;
                    max_dev = std::max(max_dev, std::abs(del_q_t));
                    x_[i].theta() += del_x_pq_[i].theta();
                    DoubleComplex const u_tmp = x_[i].v() * exp(1.0i * x_[i].theta());
                    max_dev = std::max(max_dev, cabs(u_tmp - u[i]));
                    u[i] = u_tmp;
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

        if (max_dev <= err_tol && (limit_check_countdown_ > do_limit_check) && !cache_run) {
            // force a limit check if it did not happen yet, but the solution has already converged
            limit_check_countdown_ = do_limit_check;
            return std::numeric_limits<double>::infinity();
        }
        return max_dev;
    }

    // ------------------- PV buses and Q limits (as in NewtonRaphsonPFSolver) -------------------

    // the Q row of a PV bus becomes the voltage identity V - V_set = 0; with the relative voltage
    // increment as unknown its only entry is the diagonal L = V
    void apply_pv_constraints(YBus<sym> const& y_bus) {
        IdxVector const& indptr = y_bus.row_indptr_lu();
        IdxVector const& indices = y_bus.col_indices_lu();
        for (Idx row = 0; row != n_bus_; ++row) {
            if (bus_control_[row].type != BusType::pv) {
                continue;
            }
            for (Idx k = indptr[row]; k != indptr[row + 1]; ++k) {
                Idx const column = indices[k];
                data_jac_[k].m() = RealTensor<sym>{0.0};
                if (row == column) {
                    auto const& v = x_[column].v();
                    if constexpr (is_symmetric_v<sym>) {
                        data_jac_[k].l() = v;
                    } else {
                        data_jac_[k].l() = RealTensor<asymmetric_t>{{v[0], v[1], v[2]}};
                    }
                } else {
                    data_jac_[k].l() = RealTensor<sym>{0.0};
                }
            }
            del_x_pq_[row].q() = 0.0;
        }
    }

    static LimitViolation violated_limit(RealValue<sym> const& q, double const& q_min, double const& q_max) {
        double q_total;
        if constexpr (is_symmetric_v<sym>) {
            q_total = q;
        } else {
            q_total = q(0) + q(1) + q(2);
        }

        if (!is_nan(q_max) && q_total > q_max + numerical_tolerance) {
            return LimitViolation::upper;
        }
        if (!is_nan(q_min) && q_total < q_min - numerical_tolerance) {
            return LimitViolation::lower;
        }
        return LimitViolation::none;
    }

    bool enforce_q_limits(PowerFlowInput<sym> const& input) {
        bool switched = false;
        auto const& regulators_per_load_gen = voltage_regulators_per_load_gen_.get();

        for (auto const& [bus, load_gens] : enumerated_zip_sequence(load_gens_per_bus_.get())) {
            if (bus_control_[bus].type != BusType::pv || !bus_control_[bus].q_limit.has_q_limits) {
                continue;
            }

            // Q of the regulating devices, see the derivation in NewtonRaphsonPFSolver::enforce_q_limits:
            // del_x_pq_[bus].q() = q_specified_all - q_calc_network, so
            // q_required = q_calc_network - q_specified_nonregulating = q_specified_regulating - del_x_pq_[bus].q().
            // A converter at the bus counts as non-regulating specified Q (-Q_t in del_x_pq_), so the
            // regulators also have to cover its reactive draw.
            RealValue<sym> specified_regulating_q{};
            for (Idx const load_gen : load_gens) {
                if (input.load_gen_status[load_gen] == status_off) {
                    continue;
                }
                for (Idx const regulator : regulators_per_load_gen.get_element_range(load_gen)) {
                    if (input.voltage_regulator[regulator].status != status_off) {
                        if (load_gen_type_.get()[load_gen] == LoadGenType::const_pq) {
                            specified_regulating_q += imag(input.s_injection[load_gen]);
                        } else {
                            throw UnsupportedLoadGenTypeForVoltageRegulator{std::format("{}", regulator)};
                        }
                    }
                }
            }

            RealValue<sym> const q_required = specified_regulating_q - del_x_pq_[bus].q();

            auto const limit =
                violated_limit(q_required, bus_control_[bus].q_limit.bus_q_min, bus_control_[bus].q_limit.bus_q_max);
            if (limit != LimitViolation::none) {
                for (Idx const load_gen : load_gens) {
                    for (Idx const regulator : regulators_per_load_gen.get_element_range(load_gen)) {
                        if (input.load_gen_status[load_gen] != status_off &&
                            input.voltage_regulator[regulator].status != status_off) {
                            double const q_limit_scalar = (limit == LimitViolation::upper)
                                                              ? input.voltage_regulator[regulator].q_max
                                                              : input.voltage_regulator[regulator].q_min;

                            if constexpr (is_symmetric_v<sym>) {
                                clamped_regulators_per_load_gen_[load_gen] = q_limit_scalar;
                            } else {
                                // preserve interphase scaling from specified Q
                                RealValue<asymmetric_t> const base_q = imag(input.s_injection[load_gen]);
                                double const base_q_total = base_q(0) + base_q(1) + base_q(2);

                                if (std::abs(base_q_total) > numerical_tolerance) {
                                    double const scale = q_limit_scalar / base_q_total;
                                    clamped_regulators_per_load_gen_[load_gen] = RealValue<asymmetric_t>{
                                        base_q(0) * scale, base_q(1) * scale, base_q(2) * scale};
                                } else {
                                    clamped_regulators_per_load_gen_[load_gen] = RealValue<asymmetric_t>{
                                        q_limit_scalar / 3.0, q_limit_scalar / 3.0, q_limit_scalar / 3.0};
                                }
                            }
                        }
                    }
                }

                bus_control_[bus].q_limit.limit_violated = limit;
                bus_control_[bus].type = BusType::pq;
                switched = true;
            }
        }
        return switched;
    }

    bool has_limited_q_converters() const {
        return std::ranges::any_of(vsc_, [](VscState const& vsc) {
            return vsc.param.control_q == VscQControlMode::u_ac &&
                   (!is_nan(vsc.param.q_min) || !is_nan(vsc.param.q_max));
        });
    }

    // a u_ac converter whose required Q_t leaves [q_min, q_max] is clamped and releases its voltage pin,
    // the converter analogue of the PV -> PQ switch (no switch-back)
    bool enforce_converter_q_limits() {
        bool switched = false;
        if constexpr (is_symmetric_v<sym>) {
            for (Idx bus = 0; bus != n_bus_; ++bus) {
                Idx const pin = pinning_q_vsc_of_bus_[bus];
                if (pin == -1) {
                    continue;
                }
                auto& vsc = vsc_[pin];
                auto const limit = violated_limit(vsc.q_t, vsc.param.q_min, vsc.param.q_max);
                if (limit == LimitViolation::none) {
                    continue;
                }
                vsc.q_t = limit == LimitViolation::upper ? vsc.param.q_max : vsc.param.q_min;
                pinning_q_vsc_of_bus_[bus] = -1;
                switched = true;
            }
        }
        return switched;
    }

    // ------------------- result -------------------

    // pass on values needed for distribution of reactive power among voltage regulators in calculate_pf_result
    void finalize_result(SolverOutput<sym>& output) const {
        output.bus.resize(n_bus_);
        for (Idx i = 0; i != n_bus_; ++i) {
            output.bus[i].q_limit_violated = bus_control_[i].q_limit.limit_violated;
        }
    }

    // The VSC is a zero-admittance branch, so the Y-bus injection misses its flow. Its from-side power s_f is
    // added at the AC bus before source, load_gen and regulator results are derived; the DC bus injection
    // (sum of DC line flows) already is the converter feed-in and stays as is.
    ComplexValueVector<sym> converter_bus_injection() {
        ComplexValueVector<sym> injection;
        if constexpr (is_symmetric_v<sym>) {
            if (vsc_.empty()) {
                return injection;
            }
            injection.resize(n_bus_);
            for (auto& vsc : vsc_) {
                if (vsc.param.control_p == VscControlMode::droop) {
                    // refresh with the final DC voltage (the last iterate moved U_dc once more)
                    vsc.p_t = vsc.param.droop_p_set + vsc.param.droop_gain * (vsc.param.droop_u_ref - x_[vsc.d].v());
                }
                injection[vsc.a] += DoubleComplex{vsc.p_t, vsc.q_t};
            }
        }
        return injection;
    }

    double converter_loss(VscState const& vsc, double v_a) const {
        double const i_t = std::hypot(vsc.p_t, vsc.q_t) / v_a;
        return vsc.param.loss_a + vsc.param.loss_b * i_t + vsc.param.loss_c * i_t * i_t;
    }

    // reads the final P_t as refreshed by converter_bus_injection
    void finalize_converter_output(SolverOutput<sym>& output) {
        if constexpr (is_symmetric_v<sym>) {
            for (auto const& vsc : vsc_) {
                double const v_a = x_[vsc.a].v();
                double const v_d = x_[vsc.d].v();
                double const p_f = converter_loss(vsc, v_a) - vsc.p_t;
                auto& branch = output.branch[vsc.branch];
                DoubleComplex const s_f{vsc.p_t, vsc.q_t};
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
            RealValue<sym> const& clamped_q = clamped_regulators_per_load_gen_[load_number];
            if (!is_nan(clamped_q)) {
                // load_gen hit a Q-limit (only const_pq can be regulated), use the clamped value
                del_x_pq_[bus_number].p() += real(input.s_injection[load_number]);
                del_x_pq_[bus_number].q() += clamped_q;
                continue;
            }
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
                throw BadInternalEnumValueError{"Jacobian and deviation calculation", type};
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
            auto const& s_input = input.s_injection[load_number];

            bool is_regulated = false;
            for (Idx const regulator : voltage_regulators_per_load_gen_.get().get_element_range(load_number)) {
                if (input.voltage_regulator[regulator].status != status_off &&
                    input.load_gen_status[load_number] != status_off) {
                    is_regulated = true;
                    break;
                }
            }

            // specified Q of regulated load_gens is ignored, as in NewtonRaphsonPFSolver, so that it
            // does not distort the initial angles and PQ-bus voltages
            ComplexValue<sym> const& s =
                is_regulated ? ComplexValue<sym>{RealValue<sym>{real(s_input)}, RealValue<sym>{0.0}} : s_input;
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
