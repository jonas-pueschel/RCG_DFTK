"""
multilevel rcg variant
"""

# TODO: Different interpolation for ρ ?

function interpolate_c2f(basis_c::PlaneWaveBasis{T}, basis_f::PlaneWaveBasis{T}, y) where {T}
    @assert basis_c.kgrid == basis_f.kgrid && basis_c.Ecut <= basis_f.Ecut
    Nk = length(basis_f.kpoints)
    @assert size(y)[1] == Nk

    x = Vector{Matrix{Complex{Float64}}}(undef, length(basis_f.kpoints))

    for (kpt_c, kpt_f, yk, ik) = zip(basis_c.kpoints, basis_f.kpoints, y, 1:Nk)
        @assert size(yk)[1] ==  length(G_vectors(basis_c, kpt_c))
        xk = zeros(Complex{T}, length(G_vectors(basis_f, kpt_f)), size(yk)[2])
        idx_f = 1
        for idx_c = 1:length(G_vectors(basis_f, kpt_f))
            while (kpt_c.G_vectors[idx_c] != kpt_f.G_vectors[idx_f])
                idx_f += 1
            end
            xk[idx_f, :] = yk[idx_c, :]
        end
        x[ik] = xk
    end

    return x
end

function interpolate_f2c(basis_c::PlaneWaveBasis{T}, basis_f::PlaneWaveBasis{T}, x) where {T}
    @assert basis_c.kgrid == basis_f.kgrid && basis_c.Ecut <= basis_f.Ecut
    Nk = length(basis_f.kpoints)
    @assert size(x)[1] == Nk

    y = Vector{Matrix{Complex{Float64}}}(undef, length(basis_c.kpoints))

    for (kpt_c, kpt_f, xk, ik) = zip(basis_c.kpoints, basis_f.kpoints, x, 1:Nk)
        @assert size(xk)[1] ==  length(G_vectors(basis_f, kpt_f))
        yk = zeros(Complex{T}, length(G_vectors(basis_c, kpt_c)), size(xk)[2])
        idx_f = 1
        for idx_c = 1:length(G_vectors(basis_c, kpt_c))
            while (kpt_c.G_vectors[idx_c] != kpt_f.G_vectors[idx_f])
                idx_f += 1
            end
            yk[idx_c, :] = xk[idx_f, :]
        end
        y[ik] = yk
    end

    return y
end

function π_St(x)
    Nk = size(x)[1]
    pf = []
    Sf = []
    Sfinv = []
    for ik in 1:Nk
        xk = x[ik]
        S = xk'xk
        s, U = eigen(S)
        Σ = broadcast(x -> sqrt(abs(x)), s)
        Σ_inv = broadcast(x -> 1.0 / x, Σ)
        Sfinv_k = U * Diagonal(Σ_inv) * U'
        pfk = xk * Sfinv_k
        push!(pf, pfk)
        push!(Sf, U * Diagonal(Σ) * U')
        push!(Sfinv, Sfinv_k)
    end
    return pf, Sf, Sfinv
end


function proj_TSt(ψ, v)
    Nk = size(ψ)[1]
    G = [ψ[ik]'v[ik] for ik = 1:Nk]
    G = 0.5 * [G[ik]' + G[ik] for ik = 1:Nk]
    return [v[ik] - ψ[ik] * G[ik] for ik = 1:Nk]
end


function point_reduce(basis_c::PlaneWaveBasis{T}, basis_f::PlaneWaveBasis{T}, ψ) where {T}

end

function point_prolongate(basis_c::PlaneWaveBasis{T}, basis_f::PlaneWaveBasis{T}, ϕ) where {T}

end

function invRet(p,q)
    
end

function DinvRet(p,q,v; iR = nothing)
    
end



mutable struct CoarseObjective
    basis
    res0
    ϕ0
end

function get_cost(c0::CoarseObjective, ϕ)
    
end

function get_residual(c0::CoarseObjective, ϕ)
    
end

