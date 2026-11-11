using DFTK
using RCG_DFTK
using PseudoPotentialData
using Plots
using DelimitedFiles

include("setups/multigrid_setup.jl")
include("./plotting_scripts/plot_iterates.jl")

model_name = "TiO2"  # "TiO2", "GaAs" or "silicon"
(; Ecuts, Ecuts_ref, basis_arr, basis_arr_ref, ψ_ref, basis_ref) = multigrid_setup(model_name)
tols = 0.1 * [6e-2, 2e-2, 3e-3, 1e-3, 2e-4, 3e-5]

# results go to experiments/data/<run_name>/, distribution plots to experiments/imgs/<run_name>/
run_name = "$(model_name)-error-estimators"
data_dir = "./experiments/data/$run_name/"
mkpath(data_dir); mkpath("./experiments/imgs/$run_name/")

estimator_names = ["Minv", "EAGrad", "SchurHess"]
basis_labels = ["ref", "sref"]  # full reference basis / small reference basis with extrapolation
header = ["Ecut", "Ecut_sref", "err_fine", "err_coarse"]
# times in s; Minv and EAGrad times include the residual time res_time (interpolation, density,
# Hamiltonian, residual), which refine_scfres computes internally for SchurHess
for label in basis_labels
    push!(header, "res_time_$label")
    for est in estimator_names
        push!(header, "$(est)_fine_$label", "$(est)_coarse_$label", "$(est)_time_$label")
    end
end
results = Vector{Float64}[]

ψ_iter = nothing
basis_prev = nothing

function split_error(basis, basis_ref, error; extrapolate = false)
    err_coarse = RCG_DFTK.norm_DFTK(basis, RCG_DFTK.interpolate_f2c(basis, basis_ref, error))
    if (extrapolate)
        S₀, S₁ = RCG_DFTK.shell_contributions(basis, basis_ref, error, :L2)
        if (S₀ > S₁)
            # replace the shell by the extrapolated discretization error
            err_fine = sqrt(S₀ / (1 - S₁ / S₀))
        else
            @warn "shell contribution s1 is bigger than s0"
            err_fine = sqrt(S₀ + S₁)
        end
    else
        err_tot = RCG_DFTK.norm_DFTK(basis_ref, error)
        err_fine = sqrt(err_tot^2 - err_coarse^2)
    end
    return err_fine, err_coarse
end

for (basis, basis_sref, tol) = zip(basis_arr, basis_arr_ref, tols)
    Ecut, Ecut_sref = basis.Ecut, basis_sref.Ecut
    println("\n\nEcut = $Ecut \n")
    if !isnothing(ψ_iter)
        ψ0 = RCG_DFTK.interpolate_c2f(basis_prev, basis, ψ_iter)
        ρ0 = RCG_DFTK.compute_density(basis, ψ0)
    else
        ψ0 = nothing
        ρ0 = nothing
    end

    earcg_result = h1_riemannian_conjugate_gradient(basis; ρ = ρ0, ψ = ψ0, tol)
    occupation = earcg_result.occupation

    ψ_fine = RCG_DFTK.interpolate_c2f(basis, basis_ref, earcg_result.ψ)
    ρ_fine = RCG_DFTK.compute_density(basis_ref, ψ_fine)

    coarse_ψ_sol = RCG_DFTK.interpolate_f2c(basis, basis_ref, ψ_ref)
    diff_coarse = RCG_DFTK.grassmann_diff_DFTK(earcg_result.ψ, coarse_ψ_sol)

    err_coarse_norm = RCG_DFTK.norm_DFTK(basis, diff_coarse)
    err_norm = RCG_DFTK.norm_DFTK(basis_ref, RCG_DFTK.grassmann_diff_DFTK(ψ_fine, ψ_ref))
    err_fine_norm = sqrt(err_norm^2 - err_coarse_norm^2)
    println("fine: $err_fine_norm; coarse: $err_coarse_norm")

    row = [Ecut, Ecut_sref, err_fine_norm, err_coarse_norm]
    plot_name(what) = "$run_name/$(model_name)-Ecut$(round(Int, Ecut))-$what"
    plot_diff(RCG_DFTK.grassmann_diff_DFTK(ψ_fine, ψ_ref), basis_ref, Ecuts; name = plot_name("exact"))

    for (basis_f, extrapolate, label) in zip([basis_ref, basis_sref], [false, true], basis_labels)
        # interpolate also for basis_ref (instead of reusing ψ_fine), so both timings are comparable
        res_time = @elapsed begin
            ψ_f = RCG_DFTK.interpolate_c2f(basis, basis_f, earcg_result.ψ)
            ρ_f = RCG_DFTK.compute_density(basis_f, ψ_f)
            energies, H = energy_hamiltonian(basis_f, ψ_f, occupation; ρ = ρ_f)
            Hψ, Λ, res, cost = RCG_DFTK.calculate_cost_residual(H, ψ_f, energies.total, basis_f, StandardCostResiudal())
        end
        push!(row, res_time)

        Minv_time = res_time + @elapsed begin
            Minv_err = RCG_DFTK.invert_refinement_metric_improved(basis_f, ψ_f, res)
        end
        EAGrad_time = res_time + @elapsed begin
            EAGrad_err = RCG_DFTK.calculate_gradient(ψ_f, Hψ, H, Λ, res, EAGradient(basis_f; tol = 1e-4))
        end
        SchurHess_time = @elapsed begin
            SchurHess_err = DFTK.refine_scfres(earcg_result, basis_f; tol = tol * 1e-3).δψ
        end

        for (est, est_err, est_time) in zip(estimator_names, [Minv_err, EAGrad_err, SchurHess_err],
                                            [Minv_time, EAGrad_time, SchurHess_time])
            est_fine, est_coarse = split_error(basis, basis_f, est_err; extrapolate)
            println("$est extr. $extrapolate : $est_fine , $est_coarse, $(round(est_time, sigdigits = 3)) s")
            push!(row, est_fine, est_coarse, est_time)
            plot_diff(est_err, basis_f, Ecuts; name = plot_name("$est-$label"))
        end
    end

    # rewrite the whole table after every Ecut, so an aborted run keeps the finished levels
    push!(results, row)
    open("$(data_dir)$(model_name)-results.dat", "w") do io
        println(io, join(header, " "))
        writedlm(io, permutedims(reduce(hcat, results)), ' ')
    end

    ψ_iter = earcg_result.ψ
    basis_prev = basis
end


