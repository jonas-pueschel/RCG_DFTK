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

mols_to_test = ["H2"]
#methods_to_test = ["rcg_inea_shift_greedy", "rcg_inea2_shift_greedy", "rcg_inea3_shift_greedy", "rcg_inea4_shift_greedy"]
#methods_to_test = ["rcg_inea_shift_greedy", "rcg_inea3_shift_greedy", "rcg_h1_ah", "scf", "scf_naive", "dcm_lbfgs"] # "dcm_cg"] 

#["rcg_inea_shift_greedy"]
#
# methods_to_test = ["dcm_lbfgs", "rcg_h1_greedy", "rcg_inea_shift_greedy", "rcg_h1", "rcg_h1_ah_greedy", "rcg_ea_shift", "dcm_cg", "rcg_h1_ah", "scf_naive", "rcg_ea", "rcg_inea_shift", "rcg_ea_ah_shift", "scf"]

methods_to_test = ["dcm_lbfgs", "rcg_h1_greedy", "rcg_inea_shift_greedy", "rcg_h1", "rcg_h1_ah_greedy", "rcg_ea_shift", "dcm_cg", "rcg_h1_ah", "rcg_inea_shift", "scf"]

#
#
plot_results = true
save_results = false

tol = 1e-9
init_iter = 1

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
    "H2" =>  () -> H2_setup(; Ecut = 30),
    "graphene" => () -> graphene_setup(;Ecut = 60),
    "silicon" => () -> silicon_setup(;Ecut = 60),
    "TiO2" => () -> TiO2_setup(;Ecut = 30),
    "GaAs" => () -> GaAs_setup(;Ecut = 30),
])

methods_rcg = Dict([
    "rcg_ea" => (basis, ψ1, callback) -> riemannian_conjugate_gradient(basis, ψ1; maxiter = 50, 
        callback = callback, is_converged = ResidualEvalConverged(tol, callback),
        stepsize = ExactHessianStep(2.5), gradient = EAGradient(basis, 100, RelativeΛShift(0.0)), 
        backtracking = StandardBacktracking(ArmijoRule(0.05, 0.5), 10)),
    "rcg_ea_shift" => (basis, ψ1, callback) -> riemannian_conjugate_gradient(basis, ψ1; maxiter = 75, 
        callback = callback, is_converged = ResidualEvalConverged(tol, callback),
        stepsize = ExactHessianStep(2.5), gradient = EAGradient(basis, 100, RelativeΛShift(1.1)), 
        backtracking = StandardBacktracking(ArmijoRule(0.05, 0.5), 10)),
    "rcg_ea_ah_shift" => (basis, ψ1, callback) -> riemannian_conjugate_gradient(basis, ψ1; maxiter = 75, 
        callback = callback, is_converged = ResidualEvalConverged(tol, callback),
        stepsize = ApproxHessianStep(2.5), gradient = EAGradient(basis, 100, RelativeΛShift(1.1)), 
        backtracking = StandardBacktracking(ArmijoRule(0.05, 0.5), 10)),
    "rcg_inea_shift" => (basis, ψ1, callback) -> riemannian_conjugate_gradient(basis, ψ1; maxiter = 75, 
        callback = callback, is_converged = ResidualEvalConverged(tol, callback),
        stepsize = ExactHessianStep(2.5), gradient = EAGradient(basis, 8, RelativeΛShift(1.1)), 
        backtracking = StandardBacktracking(ArmijoRule(0.05, 0.5), 1)),
    "rcg_inea_shift_greedy" => (basis, ψ1, callback) -> riemannian_conjugate_gradient(basis, ψ1; maxiter = 75, 
        callback = callback, is_converged = ResidualEvalConverged(tol, callback),
        stepsize = ExactHessianStep(2.5), gradient = EAGradient(basis, 8, RelativeΛShift(1.1)), 
        backtracking = GreedyBacktracking(ArmijoRule(0.05, 0.5), 0, 10)),
    "rcg_inea2_shift_greedy" => (basis, ψ1, callback) -> riemannian_conjugate_gradient(basis, ψ1; maxiter = 75, 
        callback = callback, is_converged = ResidualEvalConverged(tol, callback),
        stepsize = ExactHessianStep(2.5), gradient = EAGradient(basis, 8, RelativeΛShift(1.1)), 
        backtracking = GreedyBacktracking(ArmijoRule(0.05, 0.5), 0, 100)),
    "rcg_inea3_shift_greedy" => (basis, ψ1, callback) -> riemannian_conjugate_gradient(basis, ψ1; maxiter = 75, 
        callback = callback, is_converged = ResidualEvalConverged(tol, callback),
        stepsize = ExactHessianStep(2.5), gradient = EAGradient(basis, 3, RelativeΛShift(1.1)), 
        backtracking = GreedyBacktracking(ArmijoRule(0.05, 0.5), 0, 10)),
    "rcg_inea4_shift_greedy" => (basis, ψ1, callback) -> riemannian_conjugate_gradient(basis, ψ1; maxiter = 75, 
        callback = callback, is_converged = ResidualEvalConverged(tol, callback),
        stepsize = ExactHessianStep(2.5), gradient = EAGradient(basis, 3, RelativeΛShift(1.1)), 
        backtracking = GreedyBacktracking(ArmijoRule(0.05, 0.5), 0, 100)),
    "rcg_h1" => (basis, ψ1, callback) -> riemannian_conjugate_gradient(basis, ψ1; maxiter = 75, 
        callback = callback, is_converged = ResidualEvalConverged(tol, callback),
        stepsize = ExactHessianStep(2.5), gradient = H1Gradient(basis), 
        backtracking = StandardBacktracking(ArmijoRule(0.05, 0.5), 0)),
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

X = nothing
xs = Vector{Vector{Float64}}()
ys = Vector{Float64}()

β = [1.0, 15.0, 7.0, 3.5]

mol_callbacks = nothing

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

    Nk = length(basis.kpoints)
    println(Nk)
    

    sum_hams = 0.0;
    sum_times = 0;

    for method = methods_to_test
        println("\n" * method)
        mth = startswith(method, "rcg") ? EvalRCG() : startswith(method, "scf") ? EvalSCF() : EvalPDCM();
        

        callback = ResidualEvalCallback(;defaultCallback, method = mth);
    

        DFTK.reset_timer!(DFTK.timer);
        try
            methods_rcg[method](basis, ψ1, callback);
        catch e
            println(e)
        end
        print(DFTK.timer)

        update_normalizer(hn, callback);
        mol_callbacks[method] = callback;

        #save_path = result_path * result_prefix * molecule * "_" * method * ".bson";
        #BSON.@save save_path callback
        if (startswith(method, "rcg"))
            sum_hams += callback.calls_DftHamiltonian[callback.n_iter]/ (Nk * size(ψ1[1])[2])
            sum_times += callback.time_DftHamiltonian;
        end
    end

    avg_ham_time = sum_times/sum_hams

    print(avg_ham_time)



    results = [];
    #normalize
    for method = methods_to_test
        callback = mol_callbacks[method];
        result = toTestResult(callback, hn, method);
        push!(results, result);


        xss, yss = callback_to_X(callback; avg_ham_time)
        push!(xs, xss...)
        push!(ys, yss...)
    end

    c = 0
    for method = methods_to_test
        c+=1
        callback = mol_callbacks[method];
        result = results[c]
        result.hams = callback_to_vals_2(callback, β).xs
    end

    if (save_results)
        BSON.@save mol_result_path results ψ1 molecule
    end

    if (plot_results)
        display_plots(results; molecule)
    end

    legend = ""
    st1 = ""

    marks = ["diamond", "square", "triangle", "o", "asterisk", "pentagon", "Mercedes star", "|"]


    #initial residual (duplicate calculation i guess?)
    T = Float64
    print(Nk)
    ψ = deepcopy(ψ1)
    occupation = [filled_occ * ones(T, n_bands) for ik = 1:Nk]
    ρ = compute_density(basis, ψ, occupation)
    energies, H = energy_hamiltonian(basis, ψ, occupation; ρ)
    Hψ = H * ψ
    Λ = [ψ[ik]'Hψ[ik] for ik = 1:Nk]
    Λ = 0.5 * [(Λ[ik] + Λ[ik]') for ik = 1:Nk]
    res = [Hψ[ik] - ψ[ik] * Λ[ik] for ik = 1:Nk]
    init_res = norm(res)

    c = 0
    for method = methods_to_test
        c+=1
        callback = mol_callbacks[method];
        mark  = marks[c];
        color = "clr$c";
        legend_temp = replace(method, "_" => "\\_")
        legend *= "$legend_temp, "
        insert_1 = callback_to_string(callback, eval_hams = true, eval_hess = true);
        i_st1 = "
\\addplot[ %exact
color=$color,
mark=$mark,
mark repeat=1,mark phase=1
]
coordinates {(0,$init_res)$insert_1
};";
st1 *= i_st1;
    end
ymax = 2 * init_res
st = "
\\begin{figure}
    \\centering
\\begin{tikzpicture}[trim axis left]
\\begin{axis}[
width = {0.95\\textwidth},
height = 7.5cm,
        legend style={nodes={scale=0.75, transform shape}},
        ymode = log,
    xlabel={cost},
    ylabel={norm residual},
    xmax = 250,
  ymin=$tol, ymax=$ymax,
    enlargelimits=0.05,
    legend pos=north east,
    grid style=dashed,
]
$st1
\\legend{$legend};
\\end{axis}
\\end{tikzpicture} 

    \\caption{Comparison of different method for $molecule}
    \\label{fig:cmp-stepsize}
\\end{figure}
";

write("$molecule-plot.tex", st)
end
