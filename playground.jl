using LinearAlgebra

n = 8
n_c = 4
m = 3

function Ic2f(Y)
    X = zeros(Float64, n,m)
    X[1:n_c, :] = Y
    return X
end

function If2c(X)
    Y = zeros(Float64, n_c, m)
    Y[:, :] = X[1:n_c, :]
    return Y
end

function polar_ret(X)
    S = X'X
    s, U = eigen(S)
    Σ = broadcast(x -> sqrt(abs(x)), s)
    Σ_inv = broadcast(x -> 1.0 / x, Σ)
    Sfinv = U * Diagonal(Σ_inv) * U'
    X = X * Sfinv
    return X
end

proj_T(X, U) = U - 0.5 * X * (X'U + U'X)
random_tangent_vector(X) = proj_T(X, rand(Float64, size(X)...))

polar_ret(X, D) = polar_ret(X + D)
random_point() = polar_ret(rand(Float64, n, m))
random_point(n, m) = polar_ret(rand(Float64, n, m))



function inv_ret(X,Z)
    Mtx_rhs = 2 * I(m)
    Mtx_lhs = X'Z
    Yz = lyap(Mtx_lhs, -Mtx_rhs)
    return Z * Yz - X
end

function D_inv_ret(X,Z,U)
    Mtx_lhs = X'Z

    Mtx_rhs_z = 2 * I(size(X)[2])
    Yz = lyap(Mtx_lhs, -Mtx_rhs_z)

    Mtx_rhs_u = X'U * Yz
    Mtx_rhs_u = Mtx_rhs_u + Mtx_rhs_u'
    Yu = lyap(Mtx_lhs, Mtx_rhs_u)
    return Z * Yu + U * Yz
end

function adj_D_inv_ret(X,Z,U)
    Mtx_lhs = X'Z

    Mtx_rhs_z = 2 * I(size(X)[2])
    Yz = lyap(Mtx_lhs, -Mtx_rhs_z)

    Mtx_rhs_u = Z'U
    Mtx_rhs_u = Mtx_rhs_u + Mtx_rhs_u'
    Xzu = lyap(Mtx_lhs', - Mtx_rhs_u)
    W = (U - X * Xzu)* Yz
    G = W'Z
    G = G + G'
    return W - 0.5 *Z * G  
end

function r(X)
    Y = If2c(X)
    return polar_ret(Y)
end

function Dr(X, V)
    W = If2c(V)
    Y =  If2c(X)

    S = Y'Y
    s, U = eigen(S)
    Σ = broadcast(x -> sqrt(abs(x)), s)
    Σ_inv = broadcast(x -> 1.0 / x, Σ)
    Sf = U * Diagonal(Σ) * U'
    Sfinv = U * Diagonal(Σ_inv) * U'

    Mtx_rhs = Y'W
    Mtx_rhs += Mtx_rhs'
    Xv = lyap(Sf, - Mtx_rhs)

    return (W - Y * Sfinv * Xv) * Sfinv
end

function Dr_psinv(X, W)
    Y = If2c(X)

    S = Y'Y
    s, U = eigen(S)
    Σ = broadcast(x -> sqrt(abs(x)), s)
    Σ_inv = broadcast(x -> 1.0 / x, Σ)
    Sf = U * Diagonal(Σ) * U'
    Sfinv = U * Diagonal(Σ_inv) * U'

    Mtx_rhs = Sfinv * (W'Y)

    Mtx_rhs += Mtx_rhs'

    println(norm(Mtx_rhs))

    Yw = lyap(Sf, -Mtx_rhs)

    return Ic2f(W)* Sf
end 

function adj_Dr_psinv(X, V)
    
end

X = random_point()
Y = r(X)

V = random_tangent_vector(X)
W = random_tangent_vector(Y)
W2 = random_tangent_vector(X)
V2 = random_tangent_vector(Y)

S = If2c(X)'If2c(X)
s, U = eigen(S)
Σ = broadcast(x -> sqrt(abs(x)), s)
Σ_inv = broadcast(x -> 1.0 / x, Σ)
Sf = U * Diagonal(Σ) * U'
Sfinv = U * Diagonal(Σ_inv) * U'

test(U) = Ic2f(U * Sf)
test_adj(Y, V) = proj_T(Y,If2c(V * Sf))

test(Dr(X,test(W))) - test(W)
Dr(X, test(Dr(X, V))) - Dr(X,V)
dot(Dr(X,test(W)), W2 * Sf) - dot(W,  Dr(X,test(W2)) * Sf) 
dot(test(Dr(X,V)), V2 * Sfinv) - dot(test(Dr(X,V2)), V * Sfinv)


dot(test(W),V) - dot(W, test_adj(Y, V))

norm(test_adj(Y, V)'Y + Y'test_adj(Y, V))