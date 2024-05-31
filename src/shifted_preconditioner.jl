using DFTK

import LinearAlgebra: ldiv!
import LinearAlgebra: mul!

abstract type AbstractShiftedTPA end;

mutable struct ScalarShiftedTPA{T <: Real} <: AbstractShiftedTPA
    basis::PlaneWaveBasis
    kpt::Kpoint
    kin::AbstractVector{T}  # kinetic energy of every G
    #mean_kin::Union{Nothing, Vector{T}}  # mean kinetic energy of every band
    default_shift::T  # shift
    shift::T
end

function ScalarShiftedTPA(basis::PlaneWaveBasis{T}, kpt::Kpoint) where {T}
    kinetic_term = [t for t in basis.model.term_types if t isa Kinetic]
    isempty(kinetic_term) && error("Preconditioner should be disabled when no Kinetic term is used.")

    # TODO Annoying that one has to recompute the kinetic energies here. Perhaps
    #      it's better to pass a HamiltonianBlock directly and read the computed values.
    kinetic_term = only(kinetic_term)
    kin = DFTK.kinetic_energy(kinetic_term, basis.Ecut, Gplusk_vectors_cart(basis, kpt))
    ScalarShiftedTPA{T}(basis, kpt, kin, nothing, 1.0, 0.0)
end
function ScalarShiftedTPA(ham::HamiltonianBlock; kwargs...)
    ScalarShiftedTPA(ham.basis, ham.kpoint)
end

@views function ldiv!(Y, P::ScalarShiftedTPA, R)
    ldiv!(Y, Diagonal(P.kin .+ (P.shift + P.default_shift)), R)
    Y
end
ldiv!(P::ScalarShiftedTPA, R) = ldiv!(R, P, R)
(Base.:\)(P::ScalarShiftedTPA, R) = ldiv!(P, copy(R))

# These are needed by eg direct minimization with CG
@views function mul!(Y, P::ScalarShiftedTPA, R)
    mul!(Y, Diagonal(P.kin .+ (P.shift + P.default_shift)), R)
    Y
end
(Base.:*)(P::ScalarShiftedTPA, R) = mul!(copy(R), P, R)

function update_shiftedTPA!(P::ScalarShiftedTPA{T}, σ::Float64) where {T}
    #P.mean_kin = [real(dot(x, Diagonal(P.kin), x)) for x in eachcol(X)]
    P.shift = σ
end

mutable struct MatrixShiftedTPA{T <: Real} <: AbstractShiftedTPA
    basis::PlaneWaveBasis
    kpt::Kpoint
    kin::AbstractVector{T}  # kinetic energy of every G
    Λ::AbstractMatrix 
    eigs::AbstractVector{T}
    U::AbstractMatrix 
    default_shift::T
end

function MatrixShiftedTPA(basis::PlaneWaveBasis{T}, kpt::Kpoint) where {T}
    kinetic_term = [t for t in basis.model.term_types if t isa Kinetic]
    isempty(kinetic_term) && error("Preconditioner should be disabled when no Kinetic term is used.")

    # TODO Annoying that one has to recompute the kinetic energies here. Perhaps
    #      it's better to pass a HamiltonianBlock directly and read the computed values.
    kinetic_term = only(kinetic_term)
    kin = DFTK.kinetic_energy(kinetic_term, basis.Ecut, Gplusk_vectors_cart(basis, kpt))
    MatrixShiftedTPA{T}(basis, kpt, kin, nothing, nothing)
end
function MatrixShiftedTPA(ham::HamiltonianBlock; kwargs...)
    MatrixShiftedTPA(ham.basis, ham.kpoint)
end

@views function ldiv!(Y, P::MatrixShiftedTPA, R)
    Y .= Y*P.U

    Threads.@threads for n = 1:size(Y, 2)
        Y[:, n] ./=  (P.kin .+ P.default_shift .+ P.eigs[n])
    end
    Y .= Y*P.U'

    Y
end
ldiv!(P::MatrixShiftedTPA, R) = ldiv!(R, P, R)
(Base.:\)(P::MatrixShiftedTPA, R) = ldiv!(P, copy(R))

# These are needed by eg direct minimization with CG
@views function mul!(Y, P::MatrixShiftedTPA, R)
    Y .= Y*P.U
    #TODO can we use Λ here?
    Threads.@threads for n = 1:size(Y, 2)
        Y[:, n] .*=  (P.kin .+ P.default_shift .+ P.eigs[n])
    end
    Y .= Y*P.U'
    Y
end
(Base.:*)(P::MatrixShiftedTPA, R) = mul!(copy(R), P, R)


function update_shiftedTPA!(P::MatrixShiftedTPA{T}, σ::Matrix{ComplexF64}) where {T}
    P.Λ = σ;
    if (norm(σ == 0)) 
        P.eigs = [0.0 for n = 1:size(σ, 1)]
        P.U = I((σ, 1));
    else
        P.eigs, P.U = eigen(σ);
    end
end