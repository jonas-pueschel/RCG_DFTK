using DFTK
using RCG_DFTK

function get_occ(basis)
    model = basis.model
    filled_occ = DFTK.filled_occupation(model)
    n_spin = model.n_spin_components
    n_bands = div(model.n_electrons, n_spin * filled_occ, RoundUp)
    Nk = length(basis.kpoints)
    occupation = [filled_occ * ones(Float64, n_bands) for _ in 1:Nk]
    return occupation
end

function multires(basis_arr, ψ, ρ, tols, init_norm_res, init_e; 
        gradient_functor = (basis) -> EAGradient(basis), 
        coarse_solver = (basis, ψ1, ρ1, tol, callback) -> RCG_DFTK.riemannian_conjugate_gradient(basis;
        callback, ψ = ψ1, ρ = ρ1, tol, gradient = gradient_functor(basis))
    )
    nb = length(basis_arr)
    basis_f = basis_arr[end]
    basis_c = basis_arr[1]
  
    occupation = get_occ(basis_f)

    ψ1 = RCG_DFTK.restrict_point(basis_c, basis_f, ψ, ProjectiveRestriction())
    #ρ1 = DFTK.interpolate_density(ρ, basis_f, basis_c)
    ρ1 = DFTK.compute_density(basis_c, ψ1, occupation)

    ρ_f = ρ

    ccs = []
    cb = TrackResTimeCallback(default_callback, init_norm_res, init_e)
    n_iter = 1
    for basis = basis_arr[1:end-1]
        res_temp = coarse_solver(basis, ψ1, ρ1, tols[n_iter], (info) -> nothing)
        push!(ccs, true)
        occupation = res_temp.occupation
        ψ1 = RCG_DFTK.prolongate_point(basis, basis_arr[n_iter + 1], res_temp.ψ)

        #ρ1 = DFTK.interpolate_density(res_temp.ρ, basis, basis_arr[n_iter + 1])
        ρ1 = DFTK.compute_density(basis_arr[n_iter + 1], ψ1, occupation)

        # calculate fine e and res; does not count to runtime
        ψ_f = RCG_DFTK.prolongate_point(basis, basis_f, res_temp.ψ)

        ρ_prev = ρ_f
        #ρ_f = DFTK.interpolate_density(res_temp.ρ, basis, basis_f)
        ρ_f = DFTK.compute_density(basis_f, ψ_f, occupation)
        
        time_err_start = Int(time_ns())

        #TODO: use full occupation here instead of fractional?
        energies, H = energy_hamiltonian(basis_f, ψ_f, occupation; ρ = ρ_f)
        cost = energies.total

        Hψ = [H.blocks[ik] * ψk for (ik, ψk) in enumerate(ψ_f)]
        Λ = [ψk'Hψ[ik] for (ik, ψk) in enumerate(ψ_f)]
        Λ = 0.5 * [(Λk + Λk') for (ik, Λk) in enumerate(Λ)]
        res = [Hψ[ik] - ψk * Λ[ik] for (ik, ψk) in enumerate(ψ_f)]
        norm_res = RCG_DFTK.norm_DFTK(basis_f, res)
        info = (;
            ham = H, ψ = ψ_f, res, basis = basis_f, converged = false, stage = :iterate, norm_res, ρin = ρ_prev, ρout = ρ_f, n_iter,
            energies, cost, algorithm = "MultiRes",
        )
        time_err_end = Int(time_ns())
        cb.err_time += (time_err_end - time_err_start)
        cb(info)
        n_iter += 1
    end
    res_2 = coarse_solver(
        basis_f, ψ1, ρ1, tols[end], cb 
    );
    append!(ccs, [false for i = 3:res_2.n_iter])

    result = (;coarse_corrections = ccs)
    return cb, result
end