
function ortho_polar(φk)
    S = φk'φk
    σ, U = eigen(S)
    σ = broadcast(x -> 1.0/sqrt(x), σ)
    P = U * Diagonal(σ) * U'
    ψk = deepcopy(φk)
    ψk = ψk * P
    return [ψk, P]
end

function ortho_qr(φk::ArrayType) where {ArrayType <: AbstractArray}
    Nn, Nk = size(φk)
    temp = qr(φk)
    R = convert(ArrayType, temp.R)
    Q = convert(ArrayType, temp.Q)
    return [Q[1:Nn, 1:Nk], R[1:Nk, 1:Nk]]
    #ψk = convert(ArrayType, temp.Q)
    #R = convert(ArrayType, temp.R)
    #if size(x) == size(φk)
    #    return [ψk, R]
    #else
        # Sometimes QR (but funnily not always) CUDA messes up the size here
    #    return [ψk[:, 1:size(φk, 2)], R]
    #end
end
