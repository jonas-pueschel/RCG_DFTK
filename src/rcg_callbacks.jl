"""
Default callback function for `Riemannian conjugate gradient`, which prints a convergence table.
"""


function RcgDefaultCallback(; show_time = true, show_grad_norm = false, lpadding = "")
    prev_time = nothing
    prev_cost = NaN
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
            prev_cost = NaN
            grad_head = show_grad_norm ? "   log10|G| " : ""
            grad_line = show_grad_norm ? "   ---------" : ""
            mg_head = haskey(info, :coarse_corrections) ? "   CC " : ""
            mg_line = haskey(info, :coarse_corrections) ? "   ---" : ""
            println("$(lpadding)n     Cost              log10|R| $grad_head   log10(ΔC)    log10(Δρ)   Δtime  $mg_head   calls_ham")
            println("$(lpadding)---   ---------------   ---------$grad_line   ----------   ---------   -------$mg_line   ---------")
        end
        cost = haskey(info, :cost) ? info.cost : info.energies.total
        cost = isnothing(cost) ? Inf : cost
        Δρ = isnothing(info.ρin) ? nothing : norm(info.ρout - info.ρin) * sqrt(abs(info.basis.dvol))

        tstr = " "^7
        if show_time && !isnothing(prev_time)
            tstr = @sprintf " %6s" TimerOutputs.prettytime(time_ns() - prev_time)
        end

        format_log8(e) = @sprintf "%8.2f" log10(abs(e))

        Estr = (@sprintf "%+15.12f" round(cost, sigdigits = 13))[1:15]
        if isnan(prev_cost)
            ΔC = " "^10
        else
            sign = cost < prev_cost ? "  " : "+ "
            ΔC = sign * format_log8(cost - prev_cost)
        end

        mgstr = haskey(info, :coarse_corrections) ? (info.coarse_corrections[end] ? "    ✓ " : "    ✗ ") : ""

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
        print(lpadding)
        @printf "% 3d   %s   %s%s   %s   %s   %s%s   %s" info.n_iter Estr resstr gradstr ΔC Δρstr tstr mgstr calls_hamstr
        println()
        prev_cost = cost
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
