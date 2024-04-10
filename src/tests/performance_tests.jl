using Pkg
Pkg.activate("./src")

using BSON
using Optim
using LineSearches

using Plots

using DFTK
using LinearAlgebra
using Plots

include("../rcg.jl")
include("../rcg_benchmarking.jl")
include("../setups/all_setups.jl")

result_path = "./src/tests/results/"
result_prefix = "testres_"

mols_to_test = ["TiO2"]
methods_to_test = ["rcg_h1_greedy", "rcg_h1_ah"] 

#["dcm_lbfgs", "rcg_h1_greedy", "rcg_inea_shift_greedy", "rcg_h1", "rcg_h1_ah_greedy", "rcg_ea_shift", "dcm_cg", "rcg_h1_ah", "scf_naive", "rcg_ea", "rcg_inea_shift", "rcg_ea_ah_shift", "scf"]

#
#
plot_results = true
save_results = false

tol = 1e-6
init_iter = 2

@kwdef mutable struct DftHamiltonianNormalizer
    calls_energy_hamiltonian = 0
    calls_DftHamiltonian = 0
    calls_compute_density = 0
    calls_apply_K = 0
    time_energy_hamiltonian = 0
    time_DftHamiltonian = 0
    time_compute_density = 0
    time_apply_K = 0
end

function getAdjustedHams(callback::ResidualEvalCallback, hn::DftHamiltonianNormalizer)
    adj_hams = []
    for i = 1:callback.n_iter
        c_ham = callback.calls_DftHamiltonian[i];
        c_den = callback.calls_compute_density[i];
        c_en = callback.calls_energy_hamiltonian[i];
        c_aK = callback.calls_apply_K[i];

        c = c_ham;

        c += c_den * ( (hn.calls_DftHamiltonian * hn.time_compute_density) 
                    / (hn.calls_compute_density * hn.time_DftHamiltonian));
        c += c_en * ( (hn.calls_DftHamiltonian * hn.time_energy_hamiltonian) 
                / (hn.calls_energy_hamiltonian * hn.time_DftHamiltonian));
        if  (c_aK != 0) 
            c += c_aK * ( (hn.calls_DftHamiltonian * hn.time_apply_K) 
            / (hn.calls_apply_K * hn.time_DftHamiltonian));
        end
        push!(adj_hams, c)
    end

    return adj_hams;
end

@kwdef mutable struct TestResult
    method 
    n_iter
    norm_residuals
    es
    times
    hams
    adj_hams
end

function toTestResult(callback::ResidualEvalCallback, hn::DftHamiltonianNormalizer, method)
    adj_hams = getAdjustedHams(callback, hn);
    result = TestResult(
        method,
        callback.n_iter,
        callback.norm_residuals[1:callback.n_iter],
        callback.energies[1:callback.n_iter],
        callback.times[1:callback.n_iter],
        callback.calls_DftHamiltonian[1:callback.n_iter],
        adj_hams,
    )
    return result;
end

function update_normalizer(hn::DftHamiltonianNormalizer, callback::ResidualEvalCallback)
    hn.calls_energy_hamiltonian += callback.calls_energy_hamiltonian[callback.n_iter];
    hn.calls_DftHamiltonian += callback.calls_DftHamiltonian[callback.n_iter];
    hn.calls_compute_density += callback.calls_compute_density[callback.n_iter];
    hn.calls_apply_K += callback.calls_apply_K[callback.n_iter];
    hn.time_energy_hamiltonian += callback.time_energy_hamiltonian;
    hn.time_DftHamiltonian += callback.time_DftHamiltonian;
    hn.time_compute_density += callback.time_compute_density;
    hn.time_apply_K += callback.time_apply_K;
end


mol_setups = Dict([
    "H2" =>  () -> H2_setup(; Ecut = 30),
    "graphene" => () -> graphene_setup(;Ecut = 30),
    "silicon" => () -> silicon_setup(;Ecut = 30),
    "TiO2" => () -> TiO2_setup(;Ecut = 30),
    "GaAs" => () -> GaAs_setup(;Ecut = 30),
])

methods_rcg = Dict([
    "rcg_ea" => (basis, ψ1, callback) -> riemannian_conjugate_gradient(basis, ψ1; maxiter = 50, 
        callback = callback, is_converged = ResidualEvalConverged(tol, callback),
        stepsize = ExactHessianStep(2.5), gradient = EAGradient(100, RelativeEigsShift(0.0)), 
        backtracking = StandardBacktracking(AvgArmijoRule(0.05, 0.5), 10)),
    "rcg_ea_shift" => (basis, ψ1, callback) -> riemannian_conjugate_gradient(basis, ψ1; maxiter = 75, 
        callback = callback, is_converged = ResidualEvalConverged(tol, callback),
        stepsize = ExactHessianStep(2.5), gradient = EAGradient(100, RelativeEigsShift(1.1)), 
        backtracking = StandardBacktracking(AvgArmijoRule(0.05, 0.5), 10)),
    "rcg_ea_ah_shift" => (basis, ψ1, callback) -> riemannian_conjugate_gradient(basis, ψ1; maxiter = 75, 
        callback = callback, is_converged = ResidualEvalConverged(tol, callback),
        stepsize = ApproxHessianStep(2.5), gradient = EAGradient(100, RelativeEigsShift(1.1)), 
        backtracking = StandardBacktracking(AvgArmijoRule(0.05, 0.5), 10)),
    "rcg_inea_shift" => (basis, ψ1, callback) -> riemannian_conjugate_gradient(basis, ψ1; maxiter = 75, 
        callback = callback, is_converged = ResidualEvalConverged(tol, callback),
        stepsize = ExactHessianStep(2.5), gradient = EAGradient(3, RelativeEigsShift(1.1)), 
        backtracking = StandardBacktracking(AvgArmijoRule(0.05, 0.5), 10)),
    "rcg_inea_shift_greedy" => (basis, ψ1, callback) -> riemannian_conjugate_gradient(basis, ψ1; maxiter = 75, 
        callback = callback, is_converged = ResidualEvalConverged(tol, callback),
        stepsize = ExactHessianStep(2.5), gradient = EAGradient(3, RelativeEigsShift(1.1)), 
        backtracking = GreedyBacktracking(AvgArmijoRule(0.1, 0.5), 10, 10)),
    "rcg_inea2_shift_greedy" => (basis, ψ1, callback) -> riemannian_conjugate_gradient(basis, ψ1; maxiter = 75, 
        callback = callback, is_converged = ResidualEvalConverged(tol, callback),
        stepsize = ExactHessianStep(2.5), gradient = EAGradient(2, RelativeEigsShift(1.1); krylov_solver = Krylov.cg), 
        backtracking = GreedyBacktracking(AvgArmijoRule(0.1, 0.5), 10, 10)),
    "rcg_inea3_shift_greedy" => (basis, ψ1, callback) -> riemannian_conjugate_gradient(basis, ψ1; maxiter = 75, 
        callback = callback, is_converged = ResidualEvalConverged(tol, callback),
        stepsize = ExactHessianStep(2.5), gradient = EAGradient(3, RelativeEigsShift(1.1); krylov_solver = Krylov.cg), 
        backtracking = GreedyBacktracking(AvgArmijoRule(0.1, 0.5), 10, 10)),
    "rcg_h1" => (basis, ψ1, callback) -> riemannian_conjugate_gradient(basis, ψ1; maxiter = 75, 
        callback = callback, is_converged = ResidualEvalConverged(tol, callback),
        stepsize = ExactHessianStep(2.5), gradient = H1Gradient(basis), 
        backtracking = StandardBacktracking(AvgArmijoRule(0.05, 0.5), 10)),
    "rcg_h1_ah" => (basis, ψ1, callback) -> riemannian_conjugate_gradient(basis, ψ1; maxiter = 75, 
        callback = callback, is_converged = ResidualEvalConverged(tol, callback),
        stepsize = ApproxHessianStep(2.5), gradient = H1Gradient(basis), 
        backtracking = StandardBacktracking(AvgArmijoRule(0.05, 0.5), 10)),
    "rcg_h1_greedy" => (basis, ψ1, callback) -> riemannian_conjugate_gradient(basis, ψ1; maxiter = 75, 
        callback = callback, is_converged = ResidualEvalConverged(tol, callback),
        stepsize = ExactHessianStep(2.5), gradient = H1Gradient(basis), 
        backtracking = GreedyBacktracking(AvgArmijoRule(0.1, 0.5), 10, 10)),
    "rcg_h1_ah_greedy" => (basis, ψ1, callback) -> riemannian_conjugate_gradient(basis, ψ1; maxiter = 75, 
        callback = callback, is_converged = ResidualEvalConverged(tol, callback),
        stepsize = ApproxHessianStep(2.5), gradient = H1Gradient(basis), 
        backtracking = GreedyBacktracking(AvgArmijoRule(0.05, 0.5), 10, 5)),
    "scf" => (basis, ψ1, callback) -> self_consistent_field(basis;  ψ = ψ1, maxiter = 100,
        callback, is_converged = is_converged = ResidualEvalConverged(tol, callback)),
    "scf_naive" => (basis, ψ1, callback) -> self_consistent_field(basis;  ψ = ψ1, maxiter = 100,
        callback, is_converged = is_converged = ResidualEvalConverged(tol, callback),
        mixing = DFTK.SimpleMixing(), nbandsalg = DFTK.FixedBands(model), solver = DFTK.scf_damping_solver()),
    "dcm_lbfgs" => (basis, ψ1, callback) -> DFTK.direct_minimization(basis; ψ = ψ1, 
        callback = callback, is_converged = ResidualEvalConverged(tol, callback),
        optim_method = Optim.LBFGS, 
        linesearch = LineSearches.HagerZhang(), #performs better than BackTracking
    ),
    "dcm_cg" => (basis, ψ1, callback) -> DFTK.direct_minimization(basis; ψ = ψ1, 
        callback = callback, is_converged = ResidualEvalConverged(tol, callback),
        optim_method = Optim.ConjugateGradient, 
        linesearch = LineSearches.HagerZhang(), #performs better than BackTracking
    ),
]);


function display_plots(results; molecule = "")
    p = plot(title = "Iterations" * " " * molecule)
    q = plot(title = "Hamiltonians" * " " * molecule)
    r = plot(title = "Times" * " " * molecule)
    for result = results
        plot!(p , 1:result.n_iter, result.norm_residuals, label = result.method)
        plot!(q, result.hams, result.norm_residuals, label = result.method)
        plot!(r, result.times, result.norm_residuals, label = result.method)
    end
    plot!(p, yscale=:log10, minorgrid=true);
    display(p);
    plot!(q, yscale=:log10, minorgrid=true);
    display(q);
    plot!(r, yscale=:log10, minorgrid=true);
    display(r);
end


model = nothing
basis = nothing
ψ1 = nothing
callback = nothing

for molecule = mols_to_test
    mol_result_path = result_path * result_prefix * molecule * ".bson";
    mol_callbacks = Dict()
    hn = DftHamiltonianNormalizer()

    model, basis = mol_setups[molecule]()

    

    filled_occ = DFTK.filled_occupation(model)
    n_spin = model.n_spin_components
    n_bands = div(model.n_electrons, n_spin * filled_occ, RoundUp)

    ψ0 = nothing
    ψ1 = nothing

    #generate admissable initial value
    while (true)
        println("trying to find initial val")
        try
            ψ0 = [DFTK.random_orbitals(basis, kpt, n_bands) for kpt in basis.kpoints];
            if (init_iter > 0)
                scfres_start = self_consistent_field(basis; ψ = ψ0 , maxiter = init_iter);
                ψ1 = DFTK.select_occupied_orbitals(basis, scfres_start.ψ, scfres_start.occupation).ψ;
            else
                ψ1 = ψ0
            end
            scfres_test_ψ1 = self_consistent_field(basis; ψ = ψ1 , maxiter = 1);
            scfres_test_ψ1 = direct_minimization(basis; ψ = ψ1 , maxiter = 1);
            break;
        catch e
            println(e)
        end
    end
    defaultCallback = RcgDefaultCallback();


    

    for method = methods_to_test
        println("\n" * method)
        mth = startswith(method, "rcg") ? EvalRCG() : startswith(method, "scf") ? EvalSCF() : EvalPDCM();
        

        callback = ResidualEvalCallback(;defaultCallback, method = mth);
    
        DFTK.reset_timer!(DFTK.timer);
        methods_rcg[method](basis, ψ1, callback);
        update_normalizer(hn, callback);
        mol_callbacks[method] = callback;

        #save_path = result_path * result_prefix * molecule * "_" * method * ".bson";
        #BSON.@save save_path callback
    end

    results = [];
    #normalize
    for method = methods_to_test
        callback = mol_callbacks[method];
        result = toTestResult(callback, hn, method);
        push!(results, result);
    end

    if (save_results)
        BSON.@save mol_result_path results ψ1 molecule
    end

    if (plot_results)
        display_plots(results; molecule)
    end
end
