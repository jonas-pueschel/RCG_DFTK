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
random_tangent_vector(X) = proj_T(X, rand(Float64, n, m))

polar_ret(X, D) = polar_ret(X + D)
random_point() = polar_ret(rand(Float64, n, m))



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
    
end

function Dr_psinv(X, U)
    
end

function adj_Dr_psinv(X, V)
    
end

X = random_point()

δX = 5 * random_tangent_vector(X)
Y = polar_ret(X + δX)

V = random_tangent_vector(X)
W = random_tangent_vector(Y)

adDirV1 = adj_D_inv_ret(X,Y,V)

#check tangent space
println("error in tangent space: $(norm(adDirV1'Y + Y'adDirV1))")

DirW = D_inv_ret(X,Y,W)
#println(norm(DirW'X + X'DirW))



# println(norm(adirV'Y + Y'adirV))
ip1 = dot(V, DirW)
ip2 = dot(adDirV1, W)
println("error in inner product: $((ip1-ip2)/norm(ip1))")


# D = random_tangent_vector(Y)

# h = 1e-8

# Yh = polar_ret(Y, h * D)

# U = inv_ret(X, Y) # indeed equal to δX
# Uh = inv_ret(X, Yh)

# ddir_invRet1 = D_inv_ret(X, Y, D)
# # ddir_invRet2 = (Uh - U)/h
# # equal to approx 0, meaning our formula is correct
# # norm(ddir_invRet1 - ddir_invRet2)

# W = random_tangent_vector(X)

# E(Z) = dot(inv_ret(X,Z),W)

# gradE(Z) = D_inv_ret(Z, X, W)

# gradE_2(Z) = - 1/(dot(X,Z)) *  proj_T(Z, dot(W, X)/(dot(X,Z))*Z - W)

# # E_simple(Z) = dot(Z, W)

# # gradE_simple(Z) = proj_T(Z, W)


# ddir1_E = dot(W, D_inv_ret(X, Y, D))
# ddir2_E = dot(gradE(Y), D)
# ddir22_E = dot(gradE_2(Y), D)
# ddir3_E = (E(Yh) - E(Y))/h


