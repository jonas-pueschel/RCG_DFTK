# Parameters and helpers for the riemannian full multigrid (RFMG)

# Prolongates ψ_c and evaluates density, Hamiltonian, energies and residual on the fine grid.
function evaluate_prolongated(basis_c, basis_f, occupation, ψ_c)
    ψ_f = prolongate_point(basis_c, basis_f, ψ_c)
    ρ_f = DFTK.compute_density(basis_f, ψ_f, occupation)
    energies, H = energy_hamiltonian(basis_f, ψ_f, occupation; ρ = ρ_f)
    Hψ, Λ, res, cost = calculate_cost_residual(H, ψ_f, energies.total, basis_f, StandardCostResiudal())
    return (; ψ_f, ρ_f, energies, cost, H, Hψ, Λ, res)
end

# Analogue of get_next_rcg for a step along a prolongated coarse direction η_f = P η_c.
# Since retraction and prolongation commute, the step is done on the coarse grid and
# then prolongated; energy and residual are evaluated on the fine grid. The returned
# named tuple contains the fields used by do_step, and additionally the coarse point ψ_c_next.
function get_next_prolongated(basis_c, basis_f, occupation, ψ_c, η_c, τ,
                              retraction::AbstractRetraction, transport::AbstractTransport)
    ψ_c_next = calculate_retraction(ψ_c, η_c, τ, retraction)
    (; ψ_f, ρ_f, energies, cost, H, Hψ, Λ, res) = evaluate_prolongated(basis_c, basis_f, occupation, ψ_c_next)

    # transport of the direction (needed for the slope), again on the coarse grid
    Tη_c = calculate_transport(ψ_c_next, η_c, η_c, τ, ψ_c, transport, retraction; is_prev_dir = true)
    Tη_next = prolongate_vector(basis_c, basis_f, ψ_c_next, ψ_f, Tη_c, ProjectionMap())

    return (; ψ_c_next, ψ_next = ψ_f, ρ_next = ρ_f, energies_next = energies, cost_next = cost,
              H_next = H, Hψ_next = Hψ, Λ_next = Λ, res_next = res, Tη_next, τ)
end

# GreedySecantStrategy that does not re-use the previous step size, i.e. every step
# starts with the initial guess τ_0 (e.g. ConstantStep(1.0))
struct RestartedGreedySecantStrategy <: IterationStrategy
    τ_0::AbstractStepSize
    c2::Real
end
function do_step(basis, ψ, η, grad, res, T_η_old, desc, Λ, H, ρ, cost, get_next,
                 backtracking::RestartedGreedySecantStrategy)
    strat = GreedySecantStrategy(backtracking.τ_0, backtracking.c2)
    return do_step(basis, ψ, η, grad, res, T_η_old, desc, Λ, H, ρ, cost, get_next, strat)
end
