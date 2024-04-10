using DFTK

import LinearAlgebra: ldiv!
import LinearAlgebra: mul!

mutable struct PreconditionerShiftedTPA{T <: Real}
    basis::PlaneWaveBasis
    kpt::Kpoint
    kin::AbstractVector{T}  # kinetic energy of every G
    mean_kin::Union{Nothing, Vector{T}}  # mean kinetic energy of every band
    shift::T  # if mean_kin is not set by `precondprep!`, this will be used for the shift
end

function PreconditionerShiftedTPA(basis::PlaneWaveBasis{T}, kpt::Kpoint) where {T}
    kinetic_term = [t for t in basis.model.term_types if t isa Kinetic]
    isempty(kinetic_term) && error("Preconditioner should be disabled when no Kinetic term is used.")

    # TODO Annoying that one has to recompute the kinetic energies here. Perhaps
    #      it's better to pass a HamiltonianBlock directly and read the computed values.
    kinetic_term = only(kinetic_term)
    kin = DFTK.kinetic_energy(kinetic_term, basis.Ecut, Gplusk_vectors_cart(basis, kpt))
    PreconditionerShiftedTPA{T}(basis, kpt, kin, nothing, 1)
end
function PreconditionerShiftedTPA(ham::HamiltonianBlock; kwargs...)
    PreconditionerShiftedTPA(ham.basis, ham.kpoint)
end

@views function ldiv!(Y, P::PreconditionerShiftedTPA, R)
    if P.mean_kin === nothing
        ldiv!(Y, Diagonal(P.kin .+ P.shift), R)
    else
        Threads.@threads for n = 1:size(Y, 2)
            Y[:, n] .= P.mean_kin[n] ./ (P.mean_kin[n] .+ P.kin .+ P.shift) .* R[:, n]
        end
    end
    Y
end
ldiv!(P::PreconditionerShiftedTPA, R) = ldiv!(R, P, R)
(Base.:\)(P::PreconditionerShiftedTPA, R) = ldiv!(P, copy(R))

# These are needed by eg direct minimization with CG
@views function mul!(Y, P::PreconditionerShiftedTPA, R)
    if P.mean_kin === nothing
        mul!(Y, Diagonal(P.kin .+ P.shift), R)
    else
        Threads.@threads for n = 1:size(Y, 2)
            Y[:, n] .= (P.mean_kin[n] .+ P.kin .+ P.shift) ./ P.mean_kin[n] .* R[:, n]
        end
    end
    Y
end
(Base.:*)(P::PreconditionerShiftedTPA, R) = mul!(copy(R), P, R)

function precondprep!(P::PreconditionerShiftedTPA, X; shift = 1)
    #P.mean_kin = [real(dot(x, Diagonal(P.kin), x)) for x in eachcol(X)]
    P.shift = shift
end