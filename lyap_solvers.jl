
# Solve the T-Sylvester equation A * R + R^T * A^T = 2*I for an upper 
# triangular matrix R required for the inverse qR-based retraction
function tsylv2solve(A)
    p = size(A,1)
    R = zeros(ComplexF64, p,p)
    R[1,1] = 1.0 / A[1,1]
    for  j = 2:p
        b = zeros(ComplexF64, j-1)
        for i = 1: j-1
            b[i] = A[j, 1:i]' * R[1:i,i]
        end
        R[1:j,j] = A[1:j, 1:j]\[-b...,1.0]
    end
    return R
end
