include("../setups/silicon_setup.jl")
using DFTK


function calculate_grad(x, r, P)
    Px = P(x)
    Pr = P(r)
    g1 = x'Px
    g2 = x'Pr
    return Pr - Px * (g1 \ g2)
end

function ret(y)
    S = y'y
    s, U = eigen(S)
    Σ = broadcast(x -> sqrt(abs(x)), s)
    Σ_inv = broadcast(x -> 1.0 / x, Σ)
    Sfinv_k = U * Diagonal(Σ_inv) * U'
    return y * Sfinv_k
end

function grad_desc_sph(A, x_0, P; maxiter = 10000, tol = 1e-6, ψ = nothing)
    proj(x,d) = d - x * x'd

    proj_ψ(y) = isnothing(ψ) ? y : y - ψ * (ψ'y)

    x = copy(x_0)
    Ax = A(x)
    λ = real(x'Ax)
    r = Ax - x * λ
    g = calculate_grad(x, r, P)
    γ = real(dot(g,g))
    d = - copy(g)
    for k = 1:maxiter
        norm_res = norm(r)
        if norm_res < tol
            break
        end
        α = min(-real(dot(d, r))/real(dot(d,A(d))), 1)
        x = ret(proj_ψ(x + α * d))
        Ax = A(x)
        λ = real(x'Ax)
        r = Ax - x * λ
        g = calculate_grad(x, r, P)
        γ_new = real(dot(g,g))
        β = max(min(γ_new/γ, (γ_new - real(dot(d,r)))/γ),0 )
        d = - g + β * proj(x, d)
        γ = γ_new
    end
    return x, λ
end
function get_virtual_gaps(;as = 10:0.1:11.4)
    virtual_gaps = []
    for a = as
        model, basis = silicon_setup(; Ecut = 30, kgrid = [4, 4, 4], supercell_size = [1, 1, 1], a);
    
        scfres = self_consistent_field(
            basis; tol = 1e-8,
            maxiter = 100
        );
    
        Pks = [DFTK.PreconditionerTPA(basis, kpt) for kpt in basis.kpoints]
        H = scfres.ham
        ψ = scfres.ψ
        virtual_gap = Inf64
        for k = 1:length(basis.kpoints)
            proj_ψ(y) = y - ψ[k] * (ψ[k]'y)
            P(y) = proj_ψ(Pks[k] \ proj_ψ(y))
            A(y) = proj_ψ(H[k] * proj_ψ(y))
            x0 = proj_ψ(rand(ComplexF64, size(ψ[k])[1], 1))
            x0 = ret(x0)
            
            x, λ = grad_desc_sph(A, x0, P; ψ =  ψ[k])
    
            virtual_gap = min(eigmin(λ) - scfres.eigenvalues[k][4], virtual_gap)
    
        end
    
        virtual_gaps = [virtual_gaps..., virtual_gap]
    end
    return virtual_gaps
end
