using Printf
using TimerOutputs
using DFTK

"""
Default callback function for `Riemannian conjugate gradient`, which prints a convergence table.
"""


function RcgDefaultCallback(; show_time=true, show_grad_norm = false)
    prev_time   = nothing
    prev_energy = NaN
    function callback(info)
        #!mpi_master() && return info  # Rest is printing => only do on master
        if info.stage == :finalize
            if (haskey(info, :converged))
                info.converged || @warn "$(info.algorithm) not converged."
            end
            return info
        end

        # TODO We should really do this properly ... this is really messy
        if info.n_iter == 1
            if (show_grad_norm)
                println("n     Energy            log10(|R|)    log10(|G|g)   log10(ΔE)   log10(Δρ)   Δtime     calls_ham")
                println("---   ---------------   ----------    -----------   ---------   ---------   -----     ---------")
            else
                println("n     Energy            log10(|R|)    log10(ΔE)   log10(Δρ)   Δtime     calls_ham")
                println("---   ---------------   ----------    ---------   ---------   -----     ---------")
            end
        end
        E    = isnothing(info.energies) ? Inf : info.energies.total
        Δρ   = isnothing(info.ρin) ? nothing : norm(info.ρout - info.ρin) * sqrt(info.basis.dvol)

        tstr = " "^9
        if show_time && !isnothing(prev_time)
            tstr = @sprintf "   % 6s" TimerOutputs.prettytime(time_ns() - prev_time)
        end

        format_log8(e) = @sprintf "%8.2f" log10(abs(e))

        Estr    = (@sprintf "%+15.12f" round(E, sigdigits=13))[1:15]
        if isnan(prev_energy)
            ΔE = " "^9
        else
            sign = E < prev_energy ? " " : "+"
            ΔE = sign * format_log8(E - prev_energy)
        end


        Δρstr   = isnothing(Δρ) ? " "^9 : " " * format_log8(Δρ)

        if (haskey(info, :norm_res) )
            resstr = !isnothing(info.norm_res) ? " " * (@sprintf "%8.2f" log10(info.norm_res)) : " "^9
        else
            resstr = " "^9
        end

        if (show_grad_norm)
            gradstr = !isnothing(info.norm_grad) ? " " * (@sprintf "%8.2f" log10(info.norm_grad)) : " "^9
        else
            gradstr = ""
        end

        if (haskey(info, :calls_ham) )
            calls_hamstr = !isnothing(info.calls_ham) ? " " * (@sprintf "%8.2f" info.calls_ham) : " "^9
        else
            calls_hamstr = " "^9
        end

        @printf "% 3d   %s   %s   %s   %s   %s   %s   %s" info.n_iter Estr resstr gradstr ΔE Δρstr tstr calls_hamstr
        println()
        prev_energy = info.energies.total
        prev_time = time_ns()

        flush(stdout)
        info
    end
end

function RcgConvergenceResidual(tolerance)
    info -> (info.norm_res < tolerance)
end

function RcgConvergenceGradient(tolerance)
    info -> (info.norm_grad < tolerance)
end