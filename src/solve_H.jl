using DFTK
include("./rcg_options.jl")

function solve_H(x0, H, b, options::GradientOptions)
    #solving H using minres
    #TODO create solver for single i in 1:Nk?
    #TODO preconditiner!
    Nk = size(x0)[1]
    x = deepcopy(x0)

    r = b - H * x
    p0 = deepcopy(r)
    s0 = H * p0
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
        if (sum(real([tr(r[ik]'r[ik]) for ik = 1:Nk])) < 1e-8 || abs(alpha) < 1e-16)
            break
        end
        p0 .= s1
        s0 = H * s1
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

function solve_H_old(x0, H, b, options::GradientOptions)
    #solving H using minres
    #todo create solver for single i in 1:Nk?
    Nk = size(x0)[1]
    x = deepcopy(x0)

    r = b - H * x
    p0 = deepcopy(r)
    s0 = H * p0
    p1 = deepcopy(p0)
    s1 = deepcopy(s0)
    for iter = 1:options.inner_iter
        p2 = deepcopy(p1)
        p1 = deepcopy(p0)
        s2 = deepcopy(s1)
        s1 = deepcopy(s0)
        alpha = [tr(r[ik]'s1[ik])/tr(s1[ik]'s1[ik]) for ik = 1:Nk];
        x = [x[ik] + alpha[ik] * p1[ik] for ik = 1:Nk];
        r = [r[ik] - alpha[ik] * s1[ik] for ik = 1:Nk]
        if (min(real([tr(r[ik]'r[ik]) for ik = 1:Nk])...) < 1e-8)
            break
        end
        p0 .= s1
        s0 = H * s1
        beta1 = [tr(s0[ik]'s1[ik])/tr(s2[ik]'s1[ik]) for ik = 1:Nk]
        p0 = [p0[ik] - beta1[ik] * p1[ik] for ik = 1:Nk]
        s0 = [s0[ik] - beta1[ik] * s1[ik] for ik = 1:Nk]
        if iter > 1
            beta2 = [tr(s0[ik]'s2[ik])/tr(s2[ik]'s2[ik]) for ik = 1:Nk]
            p0 = [p0[ik] - beta2[ik] * p2[ik] for ik = 1:Nk]
            s0 = [s0[ik] - beta2[ik] * s2[ik] for ik = 1:Nk]
        end
    end

    return x
end