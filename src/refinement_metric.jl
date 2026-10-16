"""
    invert_refinement_metric_improved(basis, ψ, res)

More efficient version of `DFTK.invert_refinement_metric`: the inverse of the
metric on the tangent space is applied directly (per band via the Schur complement
of the projected TPA preconditioner) instead of by an inner CG.
TODO: replace by `DFTK.invert_refinement_metric` once the corresponding DFTK pull
request is merged.
"""
function invert_refinement_metric_improved(basis::PlaneWaveBasis{T}, ψ, res) where {T}
    function apply_inv_M(ψk, Pk, resk)
        resk = DFTK.proj_tangent_kpt(resk, ψk)
        Pkinv_resk = similar(ψk)
        Minv_resk = similar(ψk)
        DFTK.parallel_loop_over_range(1:size(ψk, 2)) do n
            Pkinv_resk[:, n] .= resk[:,n] ./ (Pk.mean_kin[n] .+ Pk.kin)
        end
        for n = 1:size(resk, 2)
            Pkninv_ψk = similar(ψk)
            DFTK.parallel_loop_over_range(1:size(ψk, 2)) do m
                Pkninv_ψk[:, m] .= ψk[:,m] ./ (Pk.mean_kin[n] .+ Pk.kin)
            end
            Gn = ψk'Pkninv_ψk
            Minv_resk[:, n] = Pkinv_resk[:, n] - Pkninv_ψk * (Gn \ ψk'Pkinv_resk[:, n])
        end
        return Minv_resk
    end

    map(basis.kpoints, ψ, res) do kpt, ψk, resk
        Pk = PreconditionerTPA(basis, kpt)
        DFTK.precondprep!(Pk, ψk)
        apply_inv_M(ψk, Pk, resk)
    end
end
