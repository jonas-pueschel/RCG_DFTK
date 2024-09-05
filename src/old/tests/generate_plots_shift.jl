using Pkg
Pkg.activate("./src")
using Optim
using LineSearches

using DFTK
using LinearAlgebra
using Krylov
#using Plots

include("../rcg.jl")
include("../setups/all_setups.jl")
include("../rcg_benchmarking.jl")
include("./callback_to_string.jl")

shifts = [RelativeEigsShift(0.0), RelativeEigsShift(1.01), RelativeEigsShift(1.1), RelativeEigsShift(1.25), RelativeΛShift(1.01), RelativeΛShift(1.1), RelativeΛShift(1.25)]

marks = ["diamond", "square", "triangle", "o", "asterisk", "pentagon", "Mercedes star", "|"]


model, basis = graphene_setup(; Ecut = 30);

# Convergence we desire in the residual
tol = 1e-9;

filled_occ = DFTK.filled_occupation(model);
n_spin = model.n_spin_components;
n_bands = div(model.n_electrons, n_spin * filled_occ, RoundUp);


ψ0 = [DFTK.random_orbitals(basis, kpt, n_bands) for kpt in basis.kpoints];
#SCF Iterations here to 2 or 3
scfres_start = self_consistent_field(basis; ψ = ψ0, maxiter = 1);
ψ1 = DFTK.select_occupied_orbitals(basis, scfres_start.ψ, scfres_start.occupation).ψ;

#initial residual (duplicate calculation i guess?)
T = Float64
Nk = length(basis.kpoints)
ψ = deepcopy(ψ1)
occupation = [filled_occ * ones(T, n_bands) for ik = 1:Nk]
ρ = compute_density(basis, ψ, occupation)
energies, H = energy_hamiltonian(basis, ψ, occupation; ρ)
Hψ = H * ψ
Λ = [ψ[ik]'Hψ[ik] for ik = 1:Nk]
Λ = 0.5 * [(Λ[ik] + Λ[ik]') for ik = 1:Nk]
res = [Hψ[ik] - ψ[ik] * Λ[ik] for ik = 1:Nk]
init_res = norm(res)

defaultCallback = RcgDefaultCallback()

st1 = ""

c = 0;
for shift = shifts
    c += 1;
    #EARCG
    callback = ResidualEvalCallback(;defaultCallback, method = EvalRCG());
    is_converged = ResidualEvalConverged(tol, callback);

    gradient = EAGradient(basis, 25, shift); #Ham may not be pd --> minres
    stepsize = ExactHessianStep(2.5);
    backtracking = GreedyBacktracking(ArmijoRule(0.05, 0.5), 0, 1);

    DFTK.reset_timer!(DFTK.timer);
    scfres_rcg1 = riemannian_conjugate_gradient(basis, ψ1; maxiter = 40, 
            callback = callback, is_converged = is_converged,
            stepsize = stepsize, gradient = gradient, backtracking = backtracking);
    println(DFTK.timer);

    mark  = marks[c];
    color = "clr$c";
    insert_1 = callback_to_string(callback);
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

st = "
\\begin{figure}[H]
    \\centering
\\begin{tikzpicture}
\\begin{axis}[
width = {1.0\\textwidth},
height = 7cm,
        legend style={nodes={scale=0.75, transform shape}},
        ymode = log,
    xlabel={iterations},
    ylabel={norm residual},
    xmax = 30,
  ymin=$tol, ymax=1,
    enlargelimits=0.05,
    legend pos=south east,
    grid style=dashed,
]
$st1
\\legend{no shift, RelativeEigsShift(1.01), RelativeEigsShift(1.1), RelativeEigsShift(1.25), Relative\$\\Lambda\$Shift(1.01), Relative\$\\Lambda\$Shift(1.1), Relative\$\\Lambda\$Shift(1.25)};
\\end{axis}
\\end{tikzpicture}

    \\caption{Comparison of the different shifts for graphene}
    \\label{fig:cmp-shift}
\\end{figure}
";

write("plot.tex", st)