"""
    get_Ecuts(Ecut_target; Ecut_min, mode = :factor, factor = 10^0.2,
              Δ_pre = nothing, Δ_post = Δ_pre, Ecut_kink = Inf)

Increasing array of cutoffs that ends in `Ecut_target` and has all entries `≥ Ecut_min`. It is
built downwards from `Ecut_target`, so the lowest entry lies somewhere above `Ecut_min`.
- `mode = :factor`: neighboring levels differ by the constant `factor`.
- `mode = :sqrt`: arithmetic steps in `√Ecut`, of size `Δ_pre` below `Ecut_kink` and `Δ_post`
  above. The plane-wave coefficients decay like `exp(-κ|G|)` with `|G| = √(2 Ecut)`, so a step
  `Δ` in `√Ecut` reduces the discretization error by about `exp(-√2 κ Δ)` on every level;
  `Δ = log(1/r) / (√2 κ)` gives a reduction by `r`. A step crossing the kink is split
  proportionally: the levels are equidistant in
  `t(s) = min(s, s_k) / Δ_pre + max(s - s_k, 0) / Δ_post` with `s = √Ecut`, `s_k = √Ecut_kink`.
"""
function get_Ecuts(Ecut_target; Ecut_min = 9, mode = :factor, factor = 10^0.2,
                   Δ_pre = nothing, Δ_post = Δ_pre, Ecut_kink = Inf)
    Ecut_min ≤ Ecut_target || throw(ArgumentError("Ecut_min > Ecut_target"))
    if mode == :factor
        factor > 1 || throw(ArgumentError("factor must be > 1"))
        n = floor(Int, log(Ecut_target / Ecut_min) / log(factor) + 1e-10)
        return [Ecut_target / factor^k for k = n:-1:0]
    elseif mode == :sqrt
        isnothing(Δ_pre) && throw(ArgumentError("mode = :sqrt needs Δ_pre"))
        s_k = sqrt(Ecut_kink)
        t_of_s(s) = min(s, s_k) / Δ_pre + max(s - s_k, 0) / Δ_post
        s_of_t(t) = t ≤ s_k / Δ_pre ? t * Δ_pre : s_k + (t - s_k / Δ_pre) * Δ_post
        t_target = t_of_s(sqrt(Ecut_target))
        n = floor(Int, t_target - t_of_s(sqrt(Ecut_min)) + 1e-10)
        Ecuts = [s_of_t(t_target - k)^2 for k = n:-1:0]
        Ecuts[end] = Ecut_target  # remove roundoff from the transformation
        return Ecuts
    else
        throw(ArgumentError("unknown mode $mode"))
    end
end

"""
    get_Ecuts_ref(Ecuts; mode = :factor, factor = 1.1, offset = nothing)

Reference cutoffs for the error estimator, one for each entry of `Ecuts`.
- `mode = :factor`: `factor * Ecut`.
- `mode = :offset`: `(√Ecut + offset)^2`, a shell of constant width in `√Ecut` (`offset = δ/√2`
  for a width `δ` in `|G|`). With coefficients decaying like `exp(-κ|G|)`, the shell captures
  the same fraction `1 - exp(-2√2 κ offset)` of the squared discretization error on every
  level, i.e. the efficiency index is constant; `offset ≈ 0.57/κ` gives about 0.9.
"""
function get_Ecuts_ref(Ecuts; mode = :factor, factor = 1.1, offset = nothing)
    if mode == :factor
        return factor .* Ecuts
    elseif mode == :offset
        isnothing(offset) && throw(ArgumentError("mode = :offset needs offset"))
        return (sqrt.(Ecuts) .+ offset) .^ 2
    else
        throw(ArgumentError("unknown mode $mode"))
    end
end
