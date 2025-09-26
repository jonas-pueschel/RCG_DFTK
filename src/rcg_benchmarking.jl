# Structs and methods used for performance benchmarking

mutable struct ResidualEvalCallback
    norm_residuals
    energies
    calls_energy_hamiltonian
    calls_DftHamiltonian
    calls_compute_density
    calls_apply_K
    times_tot #not the best measurement but we take it...
    times
    n_iter
    defaultCallback
    method

    function ResidualEvalCallback(; method = nothing, defaultCallback = nothing)
        return new([], [], [], [], [], [], [], [], 0, defaultCallback, method)
    end
end


function (callback::ResidualEvalCallback)(info)

    #TODO there is a lot "if else" stuff going on here, maybe this can be streamlined?

    #disable timer, because residual may be calculated
    disable_timer!(DFTK.timer)

    #push energy
    if (haskey(info, :energies))
        push!(callback.energies, info.energies.total)
    end

    #push residual; if it does not exist, calculate it
    norm_res = nothing
    if (haskey(info, :norm_res))
        norm_res = info.norm_res
    elseif (haskey(info, :ham) && haskey(info, :ψ) && haskey(info, :occupation) && haskey(info, :basis))
        H = info.ham
        ψ = info.ψ
        basis = info.basis
        occupation = info.occupation
        try
            ψ = DFTK.select_occupied_orbitals(basis, ψ, occupation).ψ
            Hψ = [H.blocks[ik] * ψk for (ik, ψk) in enumerate(ψ)]
            Λ = [ψk'Hψ[ik] for (ik, ψk) in enumerate(ψ)]
            Λ = 0.5 * [(Λk + Λk') for (ik, Λk) in enumerate(Λ)]
            res = [Hψ[ik] - ψk * Λ[ik] for (ik, ψk) in enumerate(ψ)]
            norm_res = norm(res)
        catch
            # selection of occupied orbitals failed. TODO What to do?
            # do nothing?
        end
    end

    push!(callback.norm_residuals, norm_res)
    callback.n_iter += 1

    #update callback
    if (!isnothing(callback.method))
        update_callback(callback, callback.method)
    end

    #perform the default callback
    converged = false
    calls_ham = length(callback.calls_DftHamiltonian) > 0 ? callback.calls_DftHamiltonian[end] * 1.0 / (size(info.ψ)[1] * size(info.ψ[1])[2]) : 0.0
    calls_ham -= length(callback.calls_DftHamiltonian) > 1 ? callback.calls_DftHamiltonian[end - 1] * 1.0 / (size(info.ψ)[1] * size(info.ψ[1])[2]) : 0.0
    if (!isnothing(callback.defaultCallback))
        if (!haskey(info, :converged))
            #sometimes converged is not set, so we set it here. The reason may be a bug?
            info_temp = (; info..., norm_res, converged = false, calls_ham)
        else
            converged = info.converged
            info_temp = (; info..., norm_res, calls_ham)
        end
        callback.defaultCallback(info_temp)
    end

    # update n_iter and re-enable timer

    return enable_timer!(DFTK.timer)
end

function ResidualEvalConverged(tolerance, callback::ResidualEvalCallback)
    return info -> (callback.n_iter == 0 || isnothing(callback.norm_residuals[end]) ? false : callback.norm_residuals[end] < tolerance)
end

function get_key_chain_value(timer, chain, val)
    for key in chain[1:(end - 1)]
        if !haskey(timer, key)
            return 0
        end
        timer = timer[key]["inner_timers"]
    end
    if !haskey(timer, chain[end])
        return 0
    end
    return timer[chain[end]][val]
end

abstract type AbstractEvalMethod end
struct EvalRCG <: AbstractEvalMethod end
function update_callback(callback::ResidualEvalCallback, ::EvalRCG)
    if (!haskey(TimerOutputs.todict(DFTK.timer)["inner_timers"], "riemannian_conjugate_gradient"))
        #sometimes happens with H1-gradient, but no idea why TODO?
        push!(callback.calls_apply_K, 0)
        push!(callback.calls_compute_density, 0)
        push!(callback.calls_energy_hamiltonian, 0)
        push!(callback.calls_DftHamiltonian, 0)
        return
    end

    tme_ns = time_ns()
    time_tot_ns = TimerOutputs.todict(DFTK.timer)["inner_timers"]["riemannian_conjugate_gradient"]["total_time_ns"]
    push!(callback.times, tme_ns)
    push!(callback.times_tot, time_tot_ns)


    rcg_timer = TimerOutputs.todict(DFTK.timer)["inner_timers"]["riemannian_conjugate_gradient"]["inner_timers"]
    calls_DftHamiltonian = get_key_chain_value(rcg_timer, ["DftHamiltonian multiplication", "local"], "n_calls") +
        get_key_chain_value(rcg_timer, ["do_step", "DftHamiltonian multiplication", "local"], "n_calls") +
        get_key_chain_value(rcg_timer, ["do_step", "get_next_rcg", "DftHamiltonian multiplication", "local"], "n_calls") +
        get_key_chain_value(rcg_timer, ["solve_H", "apply_H", "DftHamiltonian multiplication", "local"], "n_calls") +
        get_key_chain_value(rcg_timer, ["solve_H", "DftHamiltonian multiplication", "local"], "n_calls")

    calls_energy_hamiltonian = get_key_chain_value(rcg_timer, ["energy_hamiltonian"], "n_calls")
    + get_key_chain_value(rcg_timer, ["do_step", "get_next_rcg", "energy_hamiltonian"], "n_calls")
    calls_compute_density = get_key_chain_value(rcg_timer, ["do_step", "get_next_rcg", "compute_density"], "n_calls")
    calls_apply_K = get_key_chain_value(rcg_timer, ["do_step", "apply_K"], "n_calls")
    push!(callback.calls_apply_K, calls_apply_K)
    push!(callback.calls_compute_density, calls_compute_density)
    push!(callback.calls_energy_hamiltonian, calls_energy_hamiltonian)
    return push!(callback.calls_DftHamiltonian, calls_DftHamiltonian)

    #total_time_ns
    # callback.time_DftHamiltonian = get_key_chain_value(rcg_timer, ["DftHamiltonian multiplication", "local"], "total_time_ns") +
    #     get_key_chain_value(rcg_timer, ["do_step", "DftHamiltonian multiplication", "local"], "total_time_ns") +
    #     get_key_chain_value(rcg_timer, ["do_step", "get_next_rcg", "DftHamiltonian multiplication", "local"], "total_time_ns") +
    #     get_key_chain_value(rcg_timer, ["solve_H", "apply_H", "DftHamiltonian multiplication", "local"], "total_time_ns")
    # callback.time_energy_hamiltonian = get_key_chain_value(rcg_timer, ["energy_hamiltonian"], "total_time_ns")
    #     + get_key_chain_value(rcg_timer, ["do_step", "get_next_rcg","energy_hamiltonian"], "total_time_ns")
    # callback.time_compute_density = get_key_chain_value(rcg_timer, ["do_step", "get_next_rcg", "compute_density"], "total_time_ns")
    # callback.time_apply_K =  get_key_chain_value(rcg_timer, ["do_step", "apply_K"], "total_time_ns")

end
struct EvalSCF <: AbstractEvalMethod end
function update_callback(callback::ResidualEvalCallback, ::EvalSCF)
    if (!haskey(TimerOutputs.todict(DFTK.timer)["inner_timers"], "self_consistent_field"))
        #sometimes happens with H1-gradient, but no idea why TODO?
        push!(callback.calls_apply_K, 0)
        push!(callback.calls_compute_density, 0)
        push!(callback.calls_energy_hamiltonian, 0)
        push!(callback.calls_DftHamiltonian, 0)
        return
    end

    scf_timer = TimerOutputs.todict(DFTK.timer)["inner_timers"]["self_consistent_field"]["inner_timers"]
    #TODO somehow determine the eigen function?
    #temp = haskey(scf_timer, "local") ? scf_timer["local"]["n_calls"] : 0
    temp = 0
    calls_DftHamiltonian = get_key_chain_value(scf_timer, ["LOBPCG", "DftHamiltonian multiplication", "local"], "n_calls")
    + temp
    calls_energy_hamiltonian = get_key_chain_value(scf_timer, ["energy_hamiltonian"], "n_calls")
    calls_compute_density = get_key_chain_value(scf_timer, ["compute_density"], "n_calls")
    calls_apply_K = 0

    tme_ns = time_ns()
    time_tot_ns = TimerOutputs.todict(DFTK.timer)["inner_timers"]["self_consistent_field"]["total_time_ns"]
    push!(callback.times, tme_ns)
    push!(callback.times_tot, time_tot_ns)
    push!(callback.calls_apply_K, calls_apply_K)
    push!(callback.calls_compute_density, calls_compute_density)
    push!(callback.calls_energy_hamiltonian, calls_energy_hamiltonian)
    return push!(callback.calls_DftHamiltonian, calls_DftHamiltonian)

    # callback.time_DftHamiltonian = get_key_chain_value(scf_timer, ["LOBPCG", "DftHamiltonian multiplication", "local"], "total_time_ns")
    # callback.time_energy_hamiltonian = get_key_chain_value( scf_timer, ["energy_hamiltonian"],"total_time_ns");
    # callback.time_compute_density = get_key_chain_value( scf_timer, ["energy_hamiltonian"],"total_time_ns");
    # callback.time_apply_K =  0.0;
end

struct EvalPDCM <: AbstractEvalMethod end
function update_callback(callback::ResidualEvalCallback, ::EvalPDCM)

    # direct_minimization, newton are missing the "@timing" prefix

    # if  !haskey(TimerOutputs.todict(DFTK.timer)["inner_timers"], "direct_minimization")
    #     return
    # end


    tme_ns = time_ns()
    time_tot_ns = TimerOutputs.todict(DFTK.timer)["total_time_ns"]
    push!(callback.times, tme_ns)
    push!(callback.times_tot, time_tot_ns)

    dcm_timer = TimerOutputs.todict(DFTK.timer)["inner_timers"] #["direct_minimization"]["inner_timers"]
    calls_DftHamiltonian = get_key_chain_value(dcm_timer, ["DftHamiltonian multiplication", "local"], "n_calls")
    #calls_DftHamiltonian += haskey(dcm_timer, "local") ? dcm_timer["local"]["n_calls"] : 0;
    calls_energy_hamiltonian = get_key_chain_value(dcm_timer, ["energy_hamiltonian"], "n_calls")
    calls_compute_density = get_key_chain_value(dcm_timer, ["compute_density"], "n_calls")
    calls_apply_K = 0

    push!(callback.calls_apply_K, calls_apply_K)
    push!(callback.calls_compute_density, calls_compute_density)
    push!(callback.calls_energy_hamiltonian, calls_energy_hamiltonian)
    return push!(callback.calls_DftHamiltonian, calls_DftHamiltonian)

    # callback.time_DftHamiltonian = dcm_timer["DftHamiltonian multiplication"]["total_time_ns"];
    # callback.time_energy_hamiltonian = dcm_timer["energy_hamiltonian"]["total_time_ns"];
    # callback.time_compute_density = dcm_timer["compute_density"]["total_time_ns"];
    # callback.time_apply_K = 0.0;
end


function plot_callbacks(callbacks, names, ψ1, basis)
    model = basis.model
    filled_occ = DFTK.filled_occupation(model)
    n_spin = model.n_spin_components
    n_bands = div(model.n_electrons, n_spin * filled_occ, RoundUp)
    Nk = length(basis.kpoints)
    occupation = [filled_occ * ones(Float64, n_bands) for _ in 1:Nk]

    norm_res_0 = norm(DFTK.compute_projected_gradient(basis, ψ1, occupation))

    # iterations
    plt1 = plot(; yscale = :log, ylabel = L"\|\|R^{(k)}\|\|_F", xlabel = "Iterations")
    for (cb, method_name) in zip(callbacks, names)
        resls = [norm_res_0]
        push!(resls, getfield(cb, :norm_residuals)[1:(end - 1)]...)
        its = (0:(length(resls) - 1))
        plot!(its, resls, label = method_name)
    end
    display(plt1)

    # Hamiltonians
    plt2 = plot(; yscale = :log, ylabel = L"\|\|R^{(k)}\|\|_F", xlabel = "Hamiltonians")
    for (cb, method_name) in zip(callbacks, names)
        resls = [norm_res_0]
        push!(resls, getfield(cb, :norm_residuals)[1:(end - 1)]...)
        hams = [0.0]
        push!(hams, getfield(cb, :calls_DftHamiltonian)[1:(end - 1)]...)
        hams .*= 1 / n_bands
        plot!(hams, resls, label = method_name)
    end
    display(plt2)

    # Times
    plt3 = plot(; yscale = :log, ylabel = L"\|\|R^{(k)}\|\|_F", xlabel = "CPU time in s")
    for (cb, method_name) in zip(callbacks, names)
        resls = [norm_res_0]
        push!(resls, getfield(cb, :norm_residuals)[1:(end - 1)]...)
        times = [0.0]
        push!(times, getfield(cb, :times_tot)[1:(end - 1)]...)
        times .*= 1.0e-9
        plot!(times, resls, label = method_name)
    end
    return display(plt3)
end
