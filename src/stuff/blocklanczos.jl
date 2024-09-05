#Block Lanczos

function block_lanczos(A, Σ, B, P::PreconditionerTest)
    #solve AX + XΣ = B
    n,p = size(B)
    tV_old = 0 * B
    R = copy(B)
    tV_new, H_low = ortho_qr(R)
    V = zeros(n,0)
    H_up = I(p)
    A_j = zeros(0,0)
    A_ex = zeros(0,0)
    for j = 1:4
        
        tV_new = P\tV_new


        AtV_new = A * tV_new 
        #W = A * tV_new - tV_old * H_up
        W = AtV_new - V * V' * AtV_new
        #println(norm(W-W2))
 
        
        H_mid = tV_new'AtV_new #tr(tV_new'W) * I(p)
        
        W = W - tV_new * H_mid
        tV_old = tV_new

        tV_new, R = ortho_qr(W)
        H_low = R

        H_21_ex = tV_old'A*V

        pj = size(H_low)[1]; lj = size(A_j)[2]
        H_21 = zeros(pj, lj)
        if (lj-pj+1 > 0)
            H_21[:, end - (pj -1) : end] = H_up'
            #println(norm(H_21 - H_21_ex)/norm(H_21_ex))
        end
        H_up = H_low'
        A_j = vcat(hcat(A_j  , H_21_ex'),
                   hcat(H_21_ex , H_mid))
        V = hcat(V, tV_old)

        A_ex = V'*A*V
        println(norm(A_ex - A_j)/ norm(A_j))
    end




end

n = 20
p = 3
diag_A = n+1: 2 * n
diag_Σ = 1:p
A =  diagm(diag_A)
Σ = diagm(1:p)
B = rand(n,p)

tA = diagm()
P = PreconditionerTest(diagm([1/s for s = diag_A]),zeros(p,p))
#P = PreconditionerTest(I(n), zeros(p,p))
block_lanczos(A, Σ, B, P)








struct PreconditionerTest 
    tA
    tΣ
end

@views function ldiv!(Y, P::PreconditionerTest, R)
    Y .= P.tA * Y + Y * P.tΣ

    Y
end
ldiv!(P::PreconditionerTest, R) = ldiv!(R, P, R)
(Base.:\)(P::PreconditionerTest, R) = ldiv!(P, copy(R))


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