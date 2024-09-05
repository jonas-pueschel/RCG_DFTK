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

inner_its = [250, 25, 10, 8, 7, 6, 5, 3]

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
st2 = ""
st3 = ""

c = 0;
for inner_it = inner_its
    c += 1;
    #EARCG
    callback = ResidualEvalCallback(;defaultCallback, method = EvalRCG());
    is_converged = ResidualEvalConverged(tol, callback);


    shift = RelativeΛShift(1.1); #shift Hk with -1.1 * Λk to try to make Ham pd
    gradient = EAGradient(basis, inner_it, shift); #Ham may not be pd --> minres
    stepsize = ExactHessianStep(2.5);
    if inner_it <= 5
        rep = 1
    else
        rep = 1
    end
    backtracking = GreedyBacktracking(ArmijoRule(0.05, 0.5), 10, rep);

    DFTK.reset_timer!(DFTK.timer);
    scfres_rcg1 = riemannian_conjugate_gradient(basis, ψ1; maxiter = 40, 
            callback = callback, is_converged = is_converged,
            stepsize = stepsize, gradient = gradient, backtracking = backtracking);
    println(DFTK.timer);

    mark  = marks[c];
    color = "clr$c";
    insert_1 = callback_to_string(callback);
    i_st1 = "
\\addplot[ %exactd
color=$color,
mark=$mark,
mark repeat=1,mark phase=1
]
coordinates {(0,$init_res)$insert_1
};";
    insert_2 = callback_to_string(callback; eval_hams = true, eval_hess = false)
    i_st2 = "
    \\addplot[ %exact
    color=$color,
    mark=$mark,
mark repeat=1,mark phase=1
]
coordinates {(0,$init_res)$insert_2
};";
    st1 *= i_st1;
    st2 *= i_st2;

    insert_3 = callback_to_string(callback; eval_hams = true, eval_hess = true)
    i_st3 = "
    \\addplot[ %exact
    color=$color,
    mark=$mark,
mark repeat=1,mark phase=1
]
coordinates {(0,$init_res)$insert_3
};";
    st3 *= i_st3;
end

st = "
\\begin{figure}
    \\centering
\\begin{tikzpicture}[trim axis left]
\\begin{axis}[
width = {0.95\\textwidth},
height = 7.5cm,
        legend style={nodes={scale=0.75, transform shape}},
        ymode = log,
    xlabel={iterations},
    ylabel={norm residual},
  ymin=$tol, ymax=1,
    enlargelimits=0.05,
    legend pos=north east,
    grid style=dashed,
]
$st1
\\legend{exact, 25 steps, 10 steps, 8 steps, 7 steps, 6 steps, 5 steps, 3 steps, 2 steps};
\\end{axis}
\\end{tikzpicture}\\\\

\\begin{tikzpicture}[trim axis left]
\\begin{axis}[
    width = {0.95\\textwidth},
    height = 7.5cm,
        legend style={nodes={scale=0.75, transform shape}},
        ymode = log,
    xlabel={cost (without step size)},
    ylabel={norm residual},
    xmax = 250,
  ymin=$tol, ymax=1,)
    enlargelimits=0.05,
    legend pos=north east,
    grid style=dashed,
]
$st2
\\legend{exact, 25 steps, 10 steps, 8 steps, 7 steps, 6 steps, 5 steps, 3 steps, 2 steps};
\\end{axis}
\\end{tikzpicture}\\\\

\\begin{tikzpicture}[trim axis left]
\\begin{axis}[
    width = {0.95\\textwidth},
    height = 7.5cm,
        legend style={nodes={scale=0.75, transform shape}},
        ymode = log,
    xlabel={cost},
    ylabel={norm residual},
    xmax = 250,
  ymin=$tol, ymax=1,
    enlargelimits=0.05,
    legend pos=north east,
    grid style=dashed,
]
$st3
\\legend{exact, 25 steps, 10 steps, 8 steps, 7 steps, 6 steps, 5 steps, 3 steps, 2 steps};
\\end{axis}
\\end{tikzpicture}
    \\caption{Comparison of the different inner iterations for graphene}
    \\label{fig:cmp-iter}
\\end{figure}
";

write("plot.tex", st)