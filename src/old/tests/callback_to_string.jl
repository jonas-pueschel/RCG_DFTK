function callback_to_X(callback; avg_ham_time = 1)
    hams = callback.calls_DftHamiltonian
    hesss = callback.calls_apply_K
    ess = callback.calls_energy_hamiltonian
    dss = callback.calls_compute_density
    times = callback.times
    xs = []
    ys = Vector{Float64}()

    for i = 2:callback.n_iter
        x = [
            (hams[i]- hams[i-1]) / (Nk * size(ψ1[1])[2]),
            hesss[i] - hesss[i-1],
            (ess[i] - ess[i-1]) / Nk,
            (dss[i] - dss[i-1]) / size(ψ1[1])[2]
        ]

        y = (times[i] - times[i-1])/avg_ham_time

        push!(xs, x)
        push!(ys, y)
    end
    
    return (; xs, ys)
end

function callback_to_string(callback; eval_hams = false, eval_hess = false)

    xs,ys = callback_to_vals(callback; eval_hams, eval_hess)

    st = ""

    for (x,y) in zip(xs, ys)
        st *= "($x, $y)"
    end
    
    return st
end

function  callback_to_vals_2(callback, β) 
    resls = callback.norm_residuals
    hams = callback.calls_DftHamiltonian
    hesss = callback.calls_apply_K
    ess = callback.calls_energy_hamiltonian
    dss = callback.calls_compute_density

    xs = []
    ys = []

    for i = 1:length(resls)

        res = resls[i]
        push!(ys, res)
        v = [
            hams[i]/ (Nk * size(ψ1[1])[2]), 
            hesss[i],
            ess[i]/Nk,
            dss[i]/size(ψ1[1])[2]
        ]
        push!(xs, β'v)

    end

    return (; xs, ys)
end

function  callback_to_vals(callback; eval_hams = false, eval_hess = false)


    resls = callback.norm_residuals
    hams = callback.calls_DftHamiltonian
    hesss = callback.calls_apply_K
    ess = callback.calls_energy_hamiltonian
    dss = callback.calls_compute_density

    xs = []
    ys = []

    for i = 1:length(resls)

        res = resls[i]
        push!(ys, res)
        if (eval_hams)
            ham = hams[i] / (Nk * size(ψ1[1])[2])
            if (eval_hess)

                ham += 4 * hesss[i]
            else
                #used one ham in step size!
                #does not work for greedy as intended...
                ham -= i
            end

            ham += ess[i] * 10 / (Nk)

            ham += dss[i] * 2.5 / (size(ψ1[1])[2]) # maybe  2?
            push!(xs, ham)
        else
            push!(xs, i)
        end
    end

    return (; xs, ys)
end