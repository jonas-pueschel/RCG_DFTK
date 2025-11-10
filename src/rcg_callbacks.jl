"""
Default callback function for `Riemannian conjugate gradient`, which prints a convergence table.
"""


function RcgDefaultCallback(; show_time = true, show_grad_norm = false)
    prev_time = nothing
    prev_energy = NaN
    return function callback(info)
        #!mpi_master() && return info  # Rest is printing => only do on master
        if info.stage == :finalize
            if (haskey(info, :converged))
                info.converged || @warn "$(info.algorithm) not converged."
            end
            return info
        end

        if info.n_iter == 1
            prev_time = nothing
            prev_energy = NaN
            grad_head = show_grad_norm ? "   log10|G| " : ""
            grad_line = show_grad_norm ? "   ---------" : ""
            println("n     Energy            log10|R| $grad_head   log10(ΔE)    log10(Δρ)   Δtime     calls_ham")
            println("---   ---------------   ---------$grad_line   ----------   ---------   -------   ---------")
        end
        E = isnothing(info.energies) ? Inf : info.energies.total
        Δρ = isnothing(info.ρin) ? nothing : norm(info.ρout - info.ρin) * sqrt(abs(info.basis.dvol))

        tstr = " "^7
        if show_time && !isnothing(prev_time)
            tstr = @sprintf " %6s" TimerOutputs.prettytime(time_ns() - prev_time)
        end

        format_log8(e) = @sprintf "%8.2f" log10(abs(e))

        Estr = (@sprintf "%+15.12f" round(E, sigdigits = 13))[1:15]
        if isnan(prev_energy)
            ΔE = " "^10
        else
            sign = E < prev_energy ? "  " : "+ "
            ΔE = sign * format_log8(E - prev_energy)
        end


        Δρstr = isnothing(Δρ) ? " "^9 : " " * format_log8(Δρ)

        if (haskey(info, :norm_res))
            resstr = !isnothing(info.norm_res) ? " " * (format_log8(info.norm_res)) : " "^9
        else
            resstr = " "^9
        end

        if (show_grad_norm)
            gradstr = "   " * (!isnothing(info.norm_grad) ? " " * (format_log8(info.norm_grad)) : " "^9)
        else
            gradstr = ""
        end

        if (haskey(info, :calls_ham))
            calls_hamstr = !isnothing(info.calls_ham) ? " " * (@sprintf "%8.2f" info.calls_ham) : " "^9
        else
            calls_hamstr = " "^9
        end

        @printf "% 3d   %s   %s%s   %s   %s   %s   %s" info.n_iter Estr resstr gradstr ΔE Δρstr tstr calls_hamstr
        println()
        prev_energy = info.energies.total
        prev_time = time_ns()

        flush(stdout)
        return info
    end
end

function RcgConvergenceResidual(tolerance)
    return info -> (info.norm_res < tolerance)
end

function RcgConvergenceGradient(tolerance)
    return info -> (info.norm_grad < tolerance)
end
