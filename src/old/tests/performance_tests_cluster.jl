using Pkg
Pkg.activate("./src")

using BSON
using Optim
using LineSearches

using DFTK
using LinearAlgebra

include("../rcg.jl")
include("../rcg_benchmarking.jl")
include("../setups/all_setups.jl")

result_path = "./src/tests/results/"
result_prefix = "fullres_"
partresult_prefix = "partialres_"

mols_to_test = ["graphene", "H2", "silicon", "GaAs", "TiO2"]
methods_to_test = ["dcm_lbfgs", "rcg_h1_greedy", "rcg_inea_shift_greedy", "rcg_h1", "rcg_h1_ah_greedy", "rcg_ea_shift", "dcm_cg", "rcg_h1_ah", "scf_naive", "rcg_ea", "rcg_inea_shift", "rcg_ea_ah_shift", "scf"]

#["dcm_lbfgs", "rcg_ea_shift", "rcg_inea_shift_greedy", "rcg_h1_ah_greedy", "rcg_h1_ah", "rcg_inea_shift", "scf"]
#

tol = 1e-6
init_iter = 3

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
        stepsize = ExactHessianStep(2.5), gradient = EAGradient(basis, 100, RelativeEigsShift(0.0)), 
        backtracking = StandardBacktracking(ArmijoRule(0.05, 0.5), 10)),
    "rcg_ea_shift" => (basis, ψ1, callback) -> riemannian_conjugate_gradient(basis, ψ1; maxiter = 75, 
        callback = callback, is_converged = ResidualEvalConverged(tol, callback),
        stepsize = ExactHessianStep(2.5), gradient = EAGradient(basis, 100, RelativeEigsShift(1.1)), 
        backtracking = StandardBacktracking(ArmijoRule(0.05, 0.5), 10)),
    "rcg_ea_ah_shift" => (basis, ψ1, callback) -> riemannian_conjugate_gradient(basis, ψ1; maxiter = 75, 
        callback = callback, is_converged = ResidualEvalConverged(tol, callback),
        stepsize = ApproxHessianStep(2.5), gradient = EAGradient(basis, 100, RelativeEigsShift(1.1)), 
        backtracking = StandardBacktracking(ArmijoRule(0.05, 0.5), 10)),
    "rcg_inea_shift" => (basis, ψ1, callback) -> riemannian_conjugate_gradient(basis, ψ1; maxiter = 75, 
        callback = callback, is_converged = ResidualEvalConverged(tol, callback),
        stepsize = ExactHessianStep(2.5), gradient = EAGradient(basis, 3, RelativeEigsShift(1.1)), 
        backtracking = StandardBacktracking(ArmijoRule(0.05, 0.5), 10)),
    "rcg_inea_shift_greedy" => (basis, ψ1, callback) -> riemannian_conjugate_gradient(basis, ψ1; maxiter = 75, 
        callback = callback, is_converged = ResidualEvalConverged(tol, callback),
        stepsize = ExactHessianStep(2.5), gradient = EAGradient(basis, 3, RelativeEigsShift(1.1)), 
        backtracking = GreedyBacktracking(ArmijoRule(0.1, 0.5), 10, 10)),
    "rcg_h1" => (basis, ψ1, callback) -> riemannian_conjugate_gradient(basis, ψ1; maxiter = 75, 
        callback = callback, is_converged = ResidualEvalConverged(tol, callback),
        stepsize = ExactHessianStep(2.5), gradient = H1Gradient(basis), 
        backtracking = StandardBacktracking(ArmijoRule(0.05, 0.5), 10)),
    "rcg_h1_ah" => (basis, ψ1, callback) -> riemannian_conjugate_gradient(basis, ψ1; maxiter = 75, 
        callback = callback, is_converged = ResidualEvalConverged(tol, callback),
        stepsize = ApproxHessianStep(2.5), gradient = H1Gradient(basis), 
        backtracking = StandardBacktracking(ArmijoRule(0.05, 0.5), 10)),
    "rcg_h1_greedy" => (basis, ψ1, callback) -> riemannian_conjugate_gradient(basis, ψ1; maxiter = 75, 
        callback = callback, is_converged = ResidualEvalConverged(tol, callback),
        stepsize = ExactHessianStep(2.5), gradient = H1Gradient(basis), 
        backtracking = GreedyBacktracking(ArmijoRule(0.1, 0.5), 10, 10)),
    "rcg_h1_ah_greedy" => (basis, ψ1, callback) -> riemannian_conjugate_gradient(basis, ψ1; maxiter = 75, 
        callback = callback, is_converged = ResidualEvalConverged(tol, callback),
        stepsize = ApproxHessianStep(2.5), gradient = H1Gradient(basis), 
        backtracking = GreedyBacktracking(ArmijoRule(0.05, 0.5), 10, 5)),
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

hello_world = "hello world"
 
BSON.@save "test.bson" hello_world

model = nothing
basis = nothing
ψ1 = nothing
callback = nothing

for molecule = mols_to_test
    println("\n" * molecule * ":\n")
    global model, basis, ψ1, callback

    mol_result_path = result_path * result_prefix * molecule * ".bson";

    if (isfile(mol_result_path))
        println("Results for molecule " * molecule * " already exist, skipping...");
        continue;
    end

    mol_callbacks = Dict()
    hn = DftHamiltonianNormalizer()

    model, basis = mol_setups[molecule]()

    

    filled_occ = DFTK.filled_occupation(model)
    n_spin = model.n_spin_components
    n_bands = div(model.n_electrons, n_spin * filled_occ, RoundUp)



    mol_psi_path = result_path * partresult_prefix * molecule * ".bson";

    if (!isfile(mol_psi_path))
        ψ0 = nothing
        ψ1 = nothing
        #generate admissable initial value
        for i = 1:20
            println("find ψ1...");
            try
                ψ0 = [DFTK.random_orbitals(basis, kpt, n_bands) for kpt in basis.kpoints];
                if (init_iter > 0)
                    scfres_start = self_consistent_field(basis; ψ = ψ0 , maxiter = init_iter, nbandsalg = DFTK.FixedBands(model));
                    ψ1 = DFTK.select_occupied_orbitals(basis, scfres_start.ψ, scfres_start.occupation).ψ;
                else
                    ψ1 = ψ0;
                end
                scfres_test_ψ1 = self_consistent_field(basis; ψ = ψ1 , maxiter = 1);
                println("found.");
                break;
            catch e
                println(e);
                println("failed...");
                continue;
            end
        end
        BSON.@save mol_psi_path ψ1 ψ0
    else
        println("Loading ψ1...")
        BSON.@load mol_psi_path ψ1 ψ0
    end



    defaultCallback = RcgDefaultCallback();

    for method = methods_to_test
        println("\n" * method)
        partial_result_path = result_path * partresult_prefix * molecule * "_" * method * ".bson";
        if (isfile(partial_result_path))
            println("partial result for " * molecule *", " * method * " already exists, skipping...");
            BSON.@load partial_result_path callback;
        else
            mth = startswith(method, "rcg") ? EvalRCG() : startswith(method, "scf") ? EvalSCF() : EvalPDCM();
            callback = ResidualEvalCallback(;defaultCallback, method = mth);
        
            DFTK.reset_timer!(DFTK.timer);
            methods_rcg[method](basis, ψ1, callback);
            BSON.@save partial_result_path callback;
        end
        update_normalizer(hn, callback);
        mol_callbacks[method] = callback;
    end

    results = [];
    #normalize
    for method = methods_to_test
        callback = mol_callbacks[method];
        result = toTestResult(callback, hn, method);
        push!(results, result);
    end

    BSON.@save mol_result_path results ψ1 molecule
end

