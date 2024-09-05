using Pkg
Pkg.activate("./src")
Pkg.instantiate()

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
include("./callback_to_string.jl")

result_path = "./src/tests/results/"
result_prefix = "tryoutres_"
partresult_prefix = "tryoutres_partial_"

molecules = ["graphene", "H2", "silicon", "GaAs", "TiO2"];
Ecut = 90;
no_runs = 5;
tol = 1e-9;
init_iter = 3;

#methods_to_test = ["rcg_h1"]
#methods_to_test = ["rcg_inea_shift_greedy", "rcg_inea2_shift_greedy", "rcg_inea3_shift_greedy", "rcg_inea4_shift_greedy"]
#methods_to_test = ["rcg_inea_shift_greedy", "rcg_inea3_shift_greedy", "rcg_h1_ah", "scf", "scf_naive", "dcm_lbfgs"] # "dcm_cg"] 

#["rcg_inea_shift_greedy"]
#

methods_to_test = ["dcm_lbfgs", "rcg_h1_greedy", "rcg_inea_shift_greedy", "rcg_h1", "rcg_h1_ah_greedy", "rcg_ea_shift", "dcm_cg", "rcg_h1_ah", "scf_naive", "rcg_ea", "rcg_inea_shift", "rcg_ea_ah_shift", "scf"]

# methods_to_test = ["dcm_lbfgs", "rcg_h1_greedy", "rcg_inea_shift_greedy", "rcg_h1", "rcg_h1_ah_greedy", "rcg_ea_shift", "dcm_cg", "rcg_h1_ah", "rcg_inea_shift", "scf"]

#
#
plot_results = false

#graphene
# β = [6.45275487893541e7, 3.777553889883237e8, 7.593999625726213e8, 3.455034310315922e8]
#[6.908584402106746e7, 3.905921451268101e8, 7.247847529502422e8, 3.2198261143023205e8]



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
        callback_to_vals(callback; eval_hams = true, eval_hess = true)[1][1:callback.n_iter],
        adj_hams,
    )
    return result;
end

function update_normalizer(hn::DftHamiltonianNormalizer, callback::ResidualEvalCallback)

end


mol_setups = Dict([
    "H2" =>  () -> H2_setup(; Ecut),
    "graphene" => () -> graphene_setup(;Ecut),
    "silicon" => () -> silicon_setup(;Ecut),
    "TiO2" => () -> TiO2_setup(;Ecut),
    "GaAs" => () -> GaAs_setup(;Ecut),
])

methods_rcg = Dict([
    "rcg_ea" => (basis, ψ1, callback) -> riemannian_conjugate_gradient(basis, ψ1; maxiter = 50, 
        callback = callback, is_converged = ResidualEvalConverged(tol, callback),
        stepsize = ExactHessianStep(2.5), gradient = EAGradient(basis, 100, RelativeΛShift(0.0)), 
        backtracking = StandardBacktracking(ArmijoRule(0.05, 0.5), 10)),
    "rcg_ea_shift" => (basis, ψ1, callback) -> riemannian_conjugate_gradient(basis, ψ1; maxiter = 75, 
        callback = callback, is_converged = ResidualEvalConverged(tol, callback),
        stepsize = ExactHessianStep(2.5), gradient = EAGradient(basis, 100, CorrectedRelativeΛShift(1.1)), 
        backtracking = StandardBacktracking(ArmijoRule(0.05, 0.5), 10)),
    "rcg_ea_ah_shift" => (basis, ψ1, callback) -> riemannian_conjugate_gradient(basis, ψ1; maxiter = 75, 
        callback = callback, is_converged = ResidualEvalConverged(tol, callback),
        stepsize = ApproxHessianStep(2.5), gradient = EAGradient(basis, 100, CorrectedRelativeΛShift(1.1)), 
        backtracking = StandardBacktracking(ArmijoRule(0.05, 0.5), 10)),
    "rcg_inea_shift" => (basis, ψ1, callback) -> riemannian_conjugate_gradient(basis, ψ1; maxiter = 75, 
        callback = callback, is_converged = ResidualEvalConverged(tol, callback),
        stepsize = ExactHessianStep(2.5), gradient = EAGradient(basis, 8, CorrectedRelativeΛShift(1.1)), 
        backtracking = StandardBacktracking(ArmijoRule(0.05, 0.5), 1)),
    "rcg_inea_shift_greedy" => (basis, ψ1, callback) -> riemannian_conjugate_gradient(basis, ψ1; maxiter = 75, 
        callback = callback, is_converged = ResidualEvalConverged(tol, callback),
        stepsize = ExactHessianStep(2.5), gradient = EAGradient(basis, 8, CorrectedRelativeΛShift(1.1)), 
        backtracking = GreedyBacktracking(ArmijoRule(0.05, 0.5), 0, 10)),
    "rcg_inea2_shift_greedy" => (basis, ψ1, callback) -> riemannian_conjugate_gradient(basis, ψ1; maxiter = 75, 
        callback = callback, is_converged = ResidualEvalConverged(tol, callback),
        stepsize = ExactHessianStep(2.5), gradient = EAGradient(basis, 8, CorrectedRelativeΛShift(1.1)), 
        backtracking = GreedyBacktracking(ArmijoRule(0.05, 0.5), 0, 100)),
    "rcg_inea3_shift_greedy" => (basis, ψ1, callback) -> riemannian_conjugate_gradient(basis, ψ1; maxiter = 75, 
        callback = callback, is_converged = ResidualEvalConverged(tol, callback),
        stepsize = ExactHessianStep(2.5), gradient = EAGradient(basis, 3, CorrectedRelativeΛShift(1.1)), 
        backtracking = GreedyBacktracking(ArmijoRule(0.05, 0.5), 0, 10)),
    "rcg_inea4_shift_greedy" => (basis, ψ1, callback) -> riemannian_conjugate_gradient(basis, ψ1; maxiter = 75, 
        callback = callback, is_converged = ResidualEvalConverged(tol, callback),
        stepsize = ExactHessianStep(2.5), gradient = EAGradient(basis, 3, CorrectedRelativeΛShift(1.1)), 
        backtracking = GreedyBacktracking(ArmijoRule(0.05, 0.5), 0, 100)),
    "rcg_h1" => (basis, ψ1, callback) -> riemannian_conjugate_gradient(basis, ψ1; maxiter = 175, 
        callback = callback, is_converged = ResidualEvalConverged(tol, callback),
        stepsize = ExactHessianStep(2.5), gradient = H1Gradient(basis), 
        backtracking = StandardBacktracking(ArmijoRule(0.05, 0.5), 0)),
    "rcg_h1_ah" => (basis, ψ1, callback) -> riemannian_conjugate_gradient(basis, ψ1; maxiter = 75, 
        callback = callback, is_converged = ResidualEvalConverged(tol, callback),
        stepsize = ApproxHessianStep(2.5), gradient = H1Gradient(basis), 
        backtracking = StandardBacktracking(ArmijoRule(0.05, 0.5), 0)),
    "rcg_h1_greedy" => (basis, ψ1, callback) -> riemannian_conjugate_gradient(basis, ψ1; maxiter = 75, 
        callback = callback, is_converged = ResidualEvalConverged(tol, callback),
        stepsize = ExactHessianStep(2.5), gradient = H1Gradient(basis), 
        backtracking = GreedyBacktracking(ArmijoRule(0.1, 0.5), 10, 10)),
    "rcg_h1_ah_greedy" => (basis, ψ1, callback) -> riemannian_conjugate_gradient(basis, ψ1; maxiter = 75, 
        callback = callback, is_converged = ResidualEvalConverged(tol, callback),
        stepsize = ApproxHessianStep(2.5), gradient = H1Gradient(basis), 
        backtracking = GreedyBacktracking(ArmijoRule(0.05, 0.5), 10, 5)),
    "scf" => (basis, ψ1, callback) -> DFTK.self_consistent_field(basis;  ψ = ψ1, maxiter = 100,
        callback, is_converged = is_converged = ResidualEvalConverged(tol, callback)),
    "scf_naive" => (basis, ψ1, callback) -> DFTK.self_consistent_field(basis;  ψ = ψ1, maxiter = 100,
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
Nk = nothing
callback = nothing


for molecule = molecules

X = nothing
xs = Vector{Vector{Float64}}()
ys = Vector{Float64}()

for run = 1:no_runs
    mol_callbacks = Dict()
    hn = DftHamiltonianNormalizer()

    model, basis = mol_setups[molecule]()

    

    filled_occ = DFTK.filled_occupation(model)
    n_spin = model.n_spin_components
    n_bands = div(model.n_electrons, n_spin * filled_occ, RoundUp)

    ψ1 = nothing

    mol_psi_path = result_path * partresult_prefix * "$run" * molecule * ".bson";

    #generate admissable initial value
    if (!isfile(mol_psi_path))
        ψ1 = nothing
        #generate admissable initial value
        while true
            println("find ψ1...");
            try
                if (init_iter > 0)
                    scfres_start = DFTK.self_consistent_field(basis; maxiter = init_iter, nbandsalg = DFTK.FixedBands(model));
                    ψ1 = DFTK.select_occupied_orbitals(basis, scfres_start.ψ, scfres_start.occupation).ψ;
                end
                scfres_test_ψ1 = DFTK.self_consistent_field(basis; ψ = ψ1 , maxiter = 1);
                println("found.");
                break;
            catch e
                println(e);
                println("failed...");
                continue;
            end
        end
        BSON.@save mol_psi_path ψ1
    else
        println("Loading ψ1...")
        BSON.@load mol_psi_path ψ1
    end
    defaultCallback = RcgDefaultCallback();

    Nk = length(basis.kpoints)
    println(Nk)
    

    sum_hams = 0.0;
    sum_times = 0;

    for method = methods_to_test
        println("\n" * method)
        partial_result_path = result_path * partresult_prefix * molecule * "_Ecut$Ecut" * "_run$run" * " " * method * ".bson";
        if (isfile(partial_result_path))
            println("partial result for " * molecule *", " * method * " already exists, skipping...");
            BSON.@load partial_result_path callback;
        else
            mth = startswith(method, "rcg") ? EvalRCG() : startswith(method, "scf") ? EvalSCF() : EvalPDCM();
            callback = ResidualEvalCallback(;defaultCallback, method = mth);
        
            DFTK.reset_timer!(DFTK.timer);
            try
                methods_rcg[method](basis, ψ1, callback);
            catch e
                error_msg = sprint(showerror, e)
                st = sprint((io,v) -> show(io, "text/plain", v), stacktrace(catch_backtrace()))
                @warn "Trouble doing things:\n$(error_msg)\n$(st)"
            end
            print(DFTK.timer)
            BSON.@save partial_result_path callback;
        end
        update_normalizer(hn, callback);
        mol_callbacks[method] = callback;

        if (startswith(method, "rcg"))
            sum_hams += callback.calls_DftHamiltonian[callback.n_iter]/ (Nk * size(ψ1[1])[2])
            sum_times += callback.time_DftHamiltonian;
        end
    end

    avg_ham_time = sum_times/sum_hams


    results = [];
    #normalize
    for method = methods_to_test
        println("normalizing $method...")
        callback = mol_callbacks[method];
        result = toTestResult(callback, hn, method);
        push!(results, result);


        #xss, yss = callback_to_X(callback; avg_ham_time)
        xss, yss = callback_to_X(callback)

        push!(xs, xss...)
        push!(ys, yss...)
    end

end

println("calculating β...")
X = permutedims(hcat(xs...))

Q,R = qr(X)

b = (Q' * ys)[1:size(R)[1]] 
β = R\b
save_path = result_path * result_prefix * "beta_" * molecule * "_Ecut" * "$Ecut" *  ".bson"
BSON.@save save_path β
println(molecule * " Beta: " * "$β");
end