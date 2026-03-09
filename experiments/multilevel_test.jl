using DFTK
using RCG_DFTK
using PseudoPotentialData
using LinearAlgebra
using Plots
using BSON

include("plot_ml_test.jl")
include("multires.jl")

# this script forces precompliation for all methods, ensuring comparability in runtime
# include("precompile_methods.jl");
# precompile_methods();

include("setups/silicon_setup.jl")
include("setups/GaAs_setup.jl")
include("setups/TiO2_setup.jl")


Ecuts = [10, 16, 25, 40, 63, 101, 160]
model, basis_arr = TiO2_setup(; Ecut = Ecuts, kgrid = [2,2,2]); 
model_name = "TiO2"
basis_c = basis_arr[1]
basis_f = basis_arr[end]

# Initial value
scfres_start = self_consistent_field(basis_c; tol = 1e-8);
ψ1_c = DFTK.select_occupied_orbitals(basis_c, scfres_start.ψ, scfres_start.occupation).ψ;
ρ1_c = scfres_start.ρ
ψ1 = RCG_DFTK.interpolate_c2f(basis_c, basis_f, ψ1_c)
ρ1 = DFTK.compute_density(basis_f, ψ1, get_occ(basis_f))
#DFTK.interpolate_density(ρ1_c, basis_c, basis_f)
# scfres_start = self_consistent_field(basis_f; tol = 1e-1, nbandsalg = DFTK.FixedBands(basis_f.model));
# ψ1 = DFTK.select_occupied_orbitals(basis_f, scfres_start.ψ, scfres_start.occupation).ψ;
# ρ1 = scfres_start.ρ

# scfres_ref = self_consistent_field(basis_f; tol = 1e-10);
# e_ref = scfres_ref.energies.total 

es0 ,res0 = RCG_DFTK.init_E_res(ψ1, ρ1, basis_f)

init_norm_res = RCG_DFTK.norm_DFTK(basis_f, res0)
init_e = es0.total

# Convergence tolerance
tol = 1.0e-8;
default_callback = RcgDefaultCallback()

cbs = []
ccs_arr = []
names = []
emin = 1000

result2ccs(result) = haskey(result, :coarse_corrections) ? result.coarse_corrections : nothing;

for grad_name = ["EA", "H1"]
    gradient_functor(basis) = grad_name == "H1" ? H1Gradient(basis) : EAGradient(basis);

    cbs = []
    ccs_arr = []
    names = []

    name = "$(grad_name) 7L"
    println("\n$name")

    cb = TrackResTimeCallback(default_callback, init_norm_res, init_e)
    result = multilevel_riemannian_optimization(basis_arr; ψ = ψ1, ρ = ρ1, tol,
        callback = cb,
        coarse_model_tol = 1e-2,
        coarse_cond_tol = 0.45,
        gradients = [gradient_functor(basis) for basis = basis_arr])
    println(cb.times_tot[end] / 1e9)
    push!(cbs, cb); push!(ccs_arr, result2ccs(result)); push!(names, name)


    name = "$(grad_name) 4L"
    println("\n$name")
    cb = TrackResTimeCallback(default_callback, init_norm_res, init_e)
    result = multilevel_riemannian_optimization(basis_arr[[1,3,5,7]]; ψ = ψ1, ρ = ρ1, tol,
        callback = cb,
        coarse_model_tol = 1e-2,
        coarse_cond_tol = 0.45,
        gradients = [gradient_functor(basis) for basis = basis_arr[[1,3,5,7]]])
    println(cb.times_tot[end] / 1e9)
    push!(cbs, cb); push!(ccs_arr, result2ccs(result)); push!(names, name)

    name = "$(grad_name) 3L"
    println("\n$name")
    cb = TrackResTimeCallback(default_callback, init_norm_res, init_e)
    result = multilevel_riemannian_optimization(basis_arr[[1,4,7]]; ψ = ψ1, ρ = ρ1, tol,
        callback = cb,
        coarse_model_tol = 1e-2,
        coarse_cond_tol = 0.45,
        gradients = [gradient_functor(basis) for basis = basis_arr[[1,4,7]]])
    println(cb.times_tot[end] / 1e9)
    push!(cbs, cb); push!(ccs_arr, result2ccs(result)); push!(names, name)

    name = "$(grad_name) 2L"
    println("\n$name")
    cb = TrackResTimeCallback(default_callback, init_norm_res, init_e)
    result = two_level_riemannian_optimization(basis_c, basis_f; ψ = ψ1, ρ = ρ1, tol,
        coarse_model_tol = 1e-2,
        coarse_cond_tol = 0.45,
        iteration_strat_fine = StandardBacktracking(
            ArmijoRule(0.1, 0.5),
            ConstantStep(1.0), 10
        ),
        callback = cb,
        gradient = gradient_functor(basis_f),
        gradient_c = gradient_functor(basis_c),
        );
    println(cb.times_tot[end] / 1e9)
    push!(cbs, cb); push!(ccs_arr, result2ccs(result)); push!(names, name)

    name = "$(grad_name) 4R"
    println("\n$name")
    tols = [100 * tol,  10*tol, tol]
    bsel = [3,5,7]
    cb, result = multires(basis_arr[bsel], ψ1, ρ1, tols, init_norm_res, init_e; gradient_functor)
    println(cb.times_tot[end] / 1e9)
    push!(cbs, cb); push!(ccs_arr, result2ccs(result)); push!(names, name)

    name = "$(grad_name)RCG"
    println("\n$name")
    cb = TrackResTimeCallback(default_callback, init_norm_res, init_e)
    result = RCG_DFTK.riemannian_conjugate_gradient(basis_f;
        callback = cb, ψ = ψ1, ρ = ρ1, tol, gradient = gradient_functor(basis_f));
    println(cb.times_tot[end] / 1e9)
    push!(cbs, cb); push!(ccs_arr, result2ccs(result)); push!(names, name)

    emin = min(min([min(cb.Es...)  for cb = cbs]...), emin)
    err = 1e-15
    refval = max([cbb.Es[1] for cbb = cbs]...)
    for cb = cbs
        if emin < 0
            cb.Es .+= (err - emin)
            continue
        end
        if abs(cb.Es[end]) < 1e-12
            continue
        end
        if cb.Es[1] != refval
            cb.Es .+= (refval - cb.Es[1])
        else
            cb.Es .+= (err - emin)
        end
    end

    BSON.@save "$model_name-$grad_name-results.bson" cbs ccs_arr names Ecuts

    i = 1
    for xfield = ["iter", :times_tot]
        for yfield =  [:norm_residuals, :Es]
            #generate_plot(cbs, ccs_arr, names, xfield, yfield);
            st = generate_plot_tikz(cbs, ccs_arr, names, xfield, yfield);
            io = open("$model_name-$grad_name-plt$i.tex", "w"); write(io, st); close(io)
            i += 1
        end
    end
end

begin
    #SCF
    cbs = []
    ccs_arr = []
    names = []


    name = "SCF"
    println("\n$name")
    cb = TrackResTimeCallback(default_callback, init_norm_res, init_e)
    result = self_consistent_field(basis_f;
        callback = cb, ψ = ψ1, ρ = ρ1, tol = 0.5 * tol);
    println(cb.times_tot[end] / 1e9)
    push!(cbs, cb); push!(ccs_arr, result2ccs(result)); push!(names, name)

    name = "SCF 4R"
    println("\n$name")
    tols = [100 * tol,  10*tol, tol]
    bsel = [3,5,7]
    cb, result = multires(basis_arr[bsel], ψ1, ρ1, tols, init_norm_res, init_e; 
        coarse_solver = (basis, ψ1, ρ1, tolerance, callback) ->  self_consistent_field(
            basis;
            ψ = ψ1, 
            ρ = ρ1,
            tol = tolerance * 0.5,
            callback
        ))
    println(cb.times_tot[end] / 1e9)
    push!(cbs, cb); push!(ccs_arr, result2ccs(result)); push!(names, name)

    emin = min(min([min(cb.Es...)  for cb = cbs]...), emin)
    err = 1e-15
    refval = max([cbb.Es[1] for cbb = cbs]...)
    for cb = cbs
        if emin < 0
            cb.Es .+= (err - emin)
            continue
        end
        if abs(cb.Es[end]) < 1e-12
            continue
        end
        if cb.Es[1] != refval
            cb.Es .+= (refval - cb.Es[1])
        else
            cb.Es .+= (err - emin)
        end
    end

    BSON.@save "$model_name-SCF-results.bson" cbs ccs_arr names Ecuts

    # i = 1
    # for xfield = ["iter", :times_tot]
    #     for yfield =  [:norm_residuals, :Es]
    #         #generate_plot(cbs, ccs_arr, names, xfield, yfield);
    #         st = generate_plot_tikz(cbs, ccs_arr, names, xfield, yfield);
    #         io = open("$model_name-$grad_name-plt$i.tex", "w"); write(io, st); close(io)
    #         i += 1
    #     end
    # end

end