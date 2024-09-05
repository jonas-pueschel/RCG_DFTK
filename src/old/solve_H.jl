using DFTK
using Krylov
using LinearMaps

include("./rcg_options.jl")

function solve_H( ψ0, H, b, options::GradientOptions)
    Nk = size(ψ0)[1]
    T = Base.Float64
    pack(ψ) = copy(DFTK.reinterpret_real(DFTK.pack_ψ(ψ)))
    unpack(x) = DFTK.unpack_ψ(DFTK.reinterpret_complex(x), size.(ψ0))
    unsafe_unpack(x) = DFTK.unsafe_unpack_ψ(DFTK.reinterpret_complex(x), size.(ψ0))

    x0 = pack(ψ0)
    rhs = pack(b)

    function apply_H(x)
        ξ = unsafe_unpack(x)
        return pack(H * ξ + options.shift * ξ)
    end

    J = LinearMap{T}(apply_H, size(x0, 1))

    function apply_Prec(x)
        ξ = unsafe_unpack(x)
        Pξ = [options.prec[ik] \ ξ[ik] for ik = 1:Nk]
        return pack(Pξ)
    end

    M = LinearMap{T}(apply_Prec, size(x0, 1))

    
    res = Krylov.minres(J, rhs, x0; M, itmax = Int64(options.inner_iter))
    return unpack(res[1])
end

function solve_H_naive(x0, H, b, options::GradientOptions; tol = 1e-64)
    #solving H using minres
    #TODO preconditioner!

    Nk = size(x0)[1]
    x = deepcopy(x0)

    #shift = 0.0

    r = b - H * x # - shift * x
    p0 = deepcopy(r)
    s0 = H * p0 #+ shift * p0
    p1 = deepcopy(p0)
    s1 = deepcopy(s0)
    for iter = 1:options.inner_iter
        p2 = deepcopy(p1)
        p1 = deepcopy(p0)
        s2 = deepcopy(s1)
        s1 = deepcopy(s0)
        alpha = sum([tr(r[ik]'s1[ik])/tr(s1[ik]'s1[ik]) for ik = 1:Nk]);#
        x += alpha * p1  
        r -= alpha * s1
        if (sum(real([tr(r[ik]'r[ik]) for ik = 1:Nk])) < tol || abs(alpha) < tol)
            break
        end
        p0 .= s1
        s0 = H * s1 #+ shift * s1
        beta1 = sum([tr(s0[ik]'s1[ik]) for ik = 1:Nk]) / sum([tr(s1[ik]'s1[ik]) for ik = 1:Nk])
        p0 = [p0[ik] - beta1 * p1[ik] for ik = 1:Nk]
        s0 = [s0[ik] - beta1 * s1[ik] for ik = 1:Nk]
        if iter > 1
            beta2 = sum([tr(s0[ik]'s2[ik]) for ik = 1:Nk]) / sum([tr(s2[ik]'s2[ik]) for ik = 1:Nk])
            p0 = [p0[ik] - beta2 * p2[ik] for ik = 1:Nk]
            s0 = [s0[ik] - beta2 * s2[ik] for ik = 1:Nk]
        end
    end

    return x
end