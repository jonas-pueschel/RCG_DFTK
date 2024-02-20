using Printf
using TimerOutputs

"""
Default callback function for `Riemannian conjugate gradient`, which prints a convergence table.
"""


function RcgDefaultCallback(; show_time=true)
    prev_time   = nothing
    prev_energy = NaN
    function callback(info, norm2res)

        #!mpi_master() && return info  # Rest is printing => only do on master
        if info.stage == :finalize
            info.converged || @warn "$(info.algorithm) not converged."
            return info
        end

        # TODO We should really do this properly ... this is really messy
        if info.n_iter == 1
            println("n     Energy            log10(|res|)   log10(ΔE)   log10(Δρ)   Δtime")
            println("---   ---------------   ------------   ---------   ---------   -----")

        end
        E    = isnothing(info.energies) ? Inf : info.energies.total
        Δρ   = norm(info.ρout - info.ρin) * sqrt(info.basis.dvol)

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


        Δρstr   = " " * format_log8(Δρ)

        resstr = !isnothing(norm2res) ? " " * (@sprintf "%8.2f" log10(abs(norm2res))) : " "^9

        @printf "% 3d   %s   %s   %s   %s   %s" info.n_iter Estr resstr ΔE Δρstr tstr
        println()
        prev_energy = info.energies.total
        prev_time = time_ns()

        flush(stdout)
        info
    end
end