# Structs and methods used for performance benchmarking

mutable struct ResidualEvalCallback
    norm_residuals
    energies 
    calls_energy_hamiltonian
    calls_DftHamiltonian
    calls_compute_density
    calls_apply_K
    times #not the best measurement but we take it...
    time_energy_hamiltonian 
    time_DftHamiltonian 
    time_compute_density 
    time_apply_K 
    n_iter
    defaultCallback 
    method

    function ResidualEvalCallback(;method = nothing, defaultCallback = nothing)
        new([], [], [], [], [], [], [], 0.0, 0.0, 0.0, 0.0, 0, defaultCallback, method);
    end
end



function (callback::ResidualEvalCallback)(info)    

    #TODO there is a lot "if else" stuff going on here, maybe this can be streamlined?
 
    #disable timer, because residual may be calculated
    disable_timer!(DFTK.timer);

    #push energy
    if (haskey(info, :energies))
        push!(callback.energies, info.energies.total);
    end

    #push residual; if it does not exist, calculate it
    norm_res = nothing;
    if (haskey(info, :norm_res))
        norm_res = info.norm_res;
    elseif (haskey(info, :ham) && haskey(info, :ψ) && haskey(info, :occupation) && haskey(info, :basis)) 
        H = info.ham;
        ψ = info.ψ;
        basis = info.basis;
        occupation = info.occupation;
        try
            ψ = DFTK.select_occupied_orbitals(basis, ψ, occupation).ψ;
            Hψ = [H.blocks[ik] * ψk for (ik, ψk) in enumerate(ψ)];
            Λ = [ψk'Hψ[ik] for (ik, ψk) in enumerate(ψ)];
            Λ = 0.5 * [(Λk + Λk') for (ik, Λk) in enumerate(Λ)];
            res = [Hψ[ik] - ψk * Λ[ik] for (ik, ψk) in enumerate(ψ)];
            norm_res = norm(res);
        catch
            # selection of occupied orbitals failed. TODO What to do?
            # do nothing?
        end  
    end

    push!(callback.norm_residuals, norm_res);
    callback.n_iter += 1

    #update callback
    if (!isnothing(callback.method) )
        update_callback(callback, callback.method);
    end

    #perform the default callback
    converged = false;
    calls_ham = length(callback.calls_DftHamiltonian) > 0 ? callback.calls_DftHamiltonian[end] * 1.0/ (size(info.ψ)[1] * size(info.ψ[1])[2]) : 0.0;
    calls_ham -= length(callback.calls_DftHamiltonian) > 1 ? callback.calls_DftHamiltonian[end-1]* 1.0/ (size(info.ψ)[1] * size(info.ψ[1])[2]) : 0.0;
    if (!isnothing(callback.defaultCallback))
        if (!haskey(info, :converged))
            #sometimes converged is not set, so we set it here. The reason may be a bug?
            info_temp = (;info..., norm_res, converged = false, calls_ham);
        else
            converged = info.converged;
            info_temp = (;info..., norm_res, calls_ham);
        end
        callback.defaultCallback(info_temp);
    end

    # update n_iter and re-enable timer

    enable_timer!(DFTK.timer)
end

function ResidualEvalConverged(tolerance , callback::ResidualEvalCallback)
    info -> (callback.n_iter == 0 || isnothing(callback.norm_residuals[end]) ? false : callback.norm_residuals[end] < tolerance)
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

    time_ms = TimerOutputs.todict(DFTK.timer)["inner_timers"]["riemannian_conjugate_gradient"]["total_time_ns"]
    push!(callback.times, time_ms)

    rcg_timer = TimerOutputs.todict(DFTK.timer)["inner_timers"]["riemannian_conjugate_gradient"]["inner_timers"];
    calls_DftHamiltonian = (haskey(rcg_timer, "DftHamiltonian multiplication") && haskey(rcg_timer["DftHamiltonian multiplication"]["inner_timers"], "local+kinetic")) ?  rcg_timer["DftHamiltonian multiplication"]["inner_timers"]["local+kinetic"]["n_calls"] : 0;
    calls_energy_hamiltonian = rcg_timer["energy_hamiltonian"]["n_calls"];
    calls_compute_density = rcg_timer["compute_density"]["n_calls"];
    calls_apply_K =  haskey(rcg_timer, "apply_K") ? rcg_timer["apply_K"]["n_calls"] : 0;

    push!(callback.calls_apply_K, calls_apply_K);
    push!(callback.calls_compute_density, calls_compute_density);
    push!(callback.calls_energy_hamiltonian, calls_energy_hamiltonian);
    push!(callback.calls_DftHamiltonian, calls_DftHamiltonian);

    #total_time_ns 
    callback.time_DftHamiltonian = haskey(rcg_timer, "DftHamiltonian multiplication") ? rcg_timer["DftHamiltonian multiplication"]["time_ns"] : 0;
    callback.time_energy_hamiltonian = rcg_timer["energy_hamiltonian"]["total_time_ns"];
    callback.time_compute_density = rcg_timer["compute_density"]["total_time_ns"];
    callback.time_apply_K =  haskey(rcg_timer, "apply_K") ? rcg_timer["apply_K"]["total_time_ns"] : 0;

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
    #temp = haskey(scf_timer, "local+kinetic") ? scf_timer["local+kinetic"]["n_calls"] : 0
    temp = 0
    calls_DftHamiltonian = scf_timer["LOBPCG"]["inner_timers"]["DftHamiltonian multiplication"]["inner_timers"]["local+kinetic"]["n_calls"] 
        + temp; 
    calls_energy_hamiltonian = scf_timer["energy_hamiltonian"]["n_calls"]
    calls_compute_density = scf_timer["compute_density"]["n_calls"]
    calls_apply_K = 0

    time_ms = TimerOutputs.todict(DFTK.timer)["inner_timers"]["self_consistent_field"]["total_time_ns"]
    push!(callback.times, time_ms)
    push!(callback.calls_apply_K, calls_apply_K)
    push!(callback.calls_compute_density, calls_compute_density)
    push!(callback.calls_energy_hamiltonian, calls_energy_hamiltonian)
    push!(callback.calls_DftHamiltonian, calls_DftHamiltonian)

    callback.time_DftHamiltonian = scf_timer["LOBPCG"]["inner_timers"]["DftHamiltonian multiplication"]["total_time_ns"];
    callback.time_energy_hamiltonian = scf_timer["energy_hamiltonian"]["total_time_ns"];
    callback.time_compute_density = scf_timer["compute_density"]["total_time_ns"];
    callback.time_apply_K =  0.0;
end

struct EvalPDCM <: AbstractEvalMethod end
function update_callback(callback::ResidualEvalCallback, ::EvalPDCM)

    # direct_minimization, newton are missing the "@timing" prefix

    # if  !haskey(TimerOutputs.todict(DFTK.timer)["inner_timers"], "direct_minimization")
    #     return
    # end

    
    time_ms = TimerOutputs.todict(DFTK.timer)["total_time_ns"];
    push!(callback.times, time_ms);

    dcm_timer = TimerOutputs.todict(DFTK.timer)["inner_timers"];#["direct_minimization"]["inner_timers"]
    calls_DftHamiltonian = haskey(dcm_timer, "DftHamiltonian multiplication") ? dcm_timer["DftHamiltonian multiplication"]["inner_timers"]["local+kinetic"]["n_calls"] : 0;
    #calls_DftHamiltonian += haskey(dcm_timer, "local+kinetic") ? dcm_timer["local+kinetic"]["n_calls"] : 0;
    calls_energy_hamiltonian = dcm_timer["energy_hamiltonian"]["n_calls"];
    calls_compute_density = dcm_timer["compute_density"]["n_calls"];
    calls_apply_K = 0;

    push!(callback.calls_apply_K, calls_apply_K);
    push!(callback.calls_compute_density, calls_compute_density);
    push!(callback.calls_energy_hamiltonian, calls_energy_hamiltonian);
    push!(callback.calls_DftHamiltonian, calls_DftHamiltonian);

    callback.time_DftHamiltonian = dcm_timer["DftHamiltonian multiplication"]["total_time_ns"];
    callback.time_energy_hamiltonian = dcm_timer["energy_hamiltonian"]["total_time_ns"];
    callback.time_compute_density = dcm_timer["compute_density"]["total_time_ns"];
    callback.time_apply_K = 0.0;
end
