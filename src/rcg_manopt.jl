using DFTK
using Manopt
using Manifolds



DFTK.@timing function rcg_manopt(basis::PlaneWaveBasis{T}; 
    ρ=guess_density(basis),
    ψ=nothing,
    tol=1e-6, maxiter=100,
    callback = RcgDefaultCallback(),
    is_converged = RcgConvergenceResidual(tol),
    gradient = default_EA_gradient(basis),
    retraction = RetractionPolar(),
    cg_param = ParamFR_PRP(),
    transport_η = DifferentiatedRetractionTransport(),
    transport_grad = DifferentiatedRetractionTransport(),
    backtracking = AdaptiveBacktracking(
        WolfeHZRule(0.05, 0.1, 0.5),
        ExactHessianStep(), 10),
    ) where {T}
    start_ns = time_ns()
    # setting parameters
    model = basis.model
    #@assert iszero(model.temperature)  # temperature is not yet supported
    @assert isnothing(model.εF)        # neither are computations with fixed Fermi level

    # check that there are no virtual orbitals
    filled_occ = DFTK.filled_occupation(model)
    n_spin = model.n_spin_components
    n_bands = div(model.n_electrons, n_spin * filled_occ, RoundUp)

    if !isnothing(ψ)
        @assert length(ψ) == length(basis.kpoints)
        @assert n_bands == size(ψ[1], 2)
    else 
        ψ = [DFTK.random_orbitals(basis, kpt, n_bands) for kpt in basis.kpoints];
    end

    if isnothing(ρ)
        ρ = guess_density(basis);
    end


    # number of kpoints and occupation
    Nk = length(basis.kpoints)

    occupation = [filled_occ * ones(T, n_bands) for ik = 1:Nk]

    # iterators
    n_iter = 0

    # orbitals, densities and energies to be updated along the iterations
       
    energies, H = energy_hamiltonian(basis, ψ, occupation; ρ)


    #TODO: create product Manifolds

    
end