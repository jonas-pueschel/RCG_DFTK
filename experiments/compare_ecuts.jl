using DFTK
using RCG_DFTK
using PseudoPotentialData
using Plots
using DelimitedFiles

# this script forces precompliation for all methods, ensuring comparability in runtime
include("precompile_methods.jl");
precompile_methods();

include("setups/silicon_setup.jl")
# Silicon lattice constant in Bohr
Ecuts = [10,20,30,40,50,60]
model, basis_arr = silicon_setup(; Ecut = Ecuts, kgrid = [4,4,4]);

# Convergence tolerance
tol = 1.0e-8;

n_tries = 1

cbs_earcg_gr = []
cbs_earcg_st = []
cbs_scf = []
cbs_h1rcg = []
cbs_l2rcg = []
cbss = [cbs_earcg_gr, cbs_earcg_st, cbs_h1rcg, cbs_l2rcg, cbs_scf]
default_callback = RcgDefaultCallback()

for basis in basis_arr
    # Initial value
    for j = 1:n_tries
        scfres_start = self_consistent_field(basis; tol = 0.5e-1, nbandsalg = DFTK.FixedBands(model));
        ψ1 = DFTK.select_occupied_orbitals(basis, scfres_start.ψ, scfres_start.occupation).ψ;
        ρ1 = scfres_start.ρ;


        es0 ,res0 = RCG_DFTK.init_E_res(ψ1, ρ1, basis)

        init_norm_res = RCG_DFTK.norm_DFTK(basis, res0)
        init_e = es0.total

        #default callback
        defaultCallback = RcgDefaultCallback();

        # we note that there is a discrepancy between the Time_tot we print and the sums of Δtime 
        # this is caused by the fact that time_tot is read from DFTK.timer and Δtime uses time.ns()
        # Thus, Δtime also accounts for time not "spent in" the DFTK.timer, like calculation of 
        # the residual for SCF and other overhead caused by the benchmarking tools. 


        # EARCG-Gr
        println("\nEARCG-Gr")
        callback_earcg_gr = TrackResTimeCallback(default_callback, init_norm_res, init_e)
        DFTK.reset_timer!(DFTK.timer)
        scfres_rcg1 = energy_adaptive_riemannian_conjugate_gradient(basis; ψ = ψ1, ρ = ρ1, μ = 0, tol, 
                                                                    callback = callback_earcg_gr);
        println("Time_tot (s): $((callback_earcg_gr.times_tot[end])/ 1e9)")
        push!(cbs_earcg_gr, callback_earcg_gr)

        # EARCG-St
        println("\nEARCG-St")
        callback_earcg_st = TrackResTimeCallback(default_callback, init_norm_res, init_e)
        DFTK.reset_timer!(DFTK.timer)
        scfres_rcg2 = energy_adaptive_riemannian_conjugate_gradient(basis; ψ = ψ1, ρ = ρ1, μ = 0, tol, 
                                                                    callback = callback_earcg_st);
        println("Time_tot (s): $((callback_earcg_st.times_tot[end])/ 1e9)")
        push!(cbs_earcg_st, RCG_DFTK.to_named_tuple(callback_earcg_st))

        # H1RCG
        println("\nH1RCG")
        callback_h1rcg = TrackResTimeCallback(default_callback, init_norm_res, init_e)
        DFTK.reset_timer!(DFTK.timer)
        scfres_rcg3 = h1_riemannian_conjugate_gradient(basis; ψ = ψ1, ρ = ρ1, tol,
                                                    callback = callback_h1rcg);
        println("Time_tot (s): $((callback_h1rcg.times_tot[end])/ 1e9)")
        push!(cbs_h1rcg, RCG_DFTK.to_named_tuple(callback_h1rcg))

        # L2RCG
        println("\nL2RCG")
        callback_l2rcg = TrackResTimeCallback(default_callback, init_norm_res, init_e)
        DFTK.reset_timer!(DFTK.timer)
        scfres_rcg4 = l2_riemannian_conjugate_gradient(basis; ψ = ψ1, ρ = ρ1, tol,
                                                    callback = callback_l2rcg);
        println("Time_tot (s): $((callback_l2rcg.times_tot[end])/ 1e9)")
        push!(cbs_l2rcg, RCG_DFTK.to_named_tuple(callback_l2rcg))

        # SCF
        println("\nSCF")
        callback_scf = TrackResTimeCallback(default_callback, init_norm_res, init_e)
        DFTK.reset_timer!(DFTK.timer)
        scfres_scf = self_consistent_field(basis; ψ = ψ1, ρ = ρ1, tol,
                                        callback = callback_scf);
        println("Time_tot (s): $((callback_scf.times_tot[end])/ 1e9)")
        push!(cbs_scf, RCG_DFTK.to_named_tuple(callback_scf))
    end
end

function get_iters(cbs; hams = false)
    i = 1
    a = [0.0 for e = Ecuts]
    for k = 1:length(Ecuts)
        sm = 0
        for j = 1:n_tries
            if hams
                sm += cbs[i].hams[end]
            else
                sm += length(cbs[i].norm_residuals)-2
            end
            i += 1
        end
        a[k] = sm/n_tries
    end
    return a
end

plt1 = plot(; ylabel = "iters", xlabel = "Ecut")
plt2 = plot(; ylabel = "perc", xlabel = "Ecut")
for (cbs, name) = zip(cbss, ["EARCG-Gr", "EARCG-St", "H1RCG", "L2RCG", "SCF"])
    iters = get_iters(cbs)
    perc = iters./iters[1] 
    plot!(plt1, Ecuts, iters, label = name)
    plot!(plt2, Ecuts, perc, label = name)
    open("experiments/data/ecuts_$(name)_iter.dat", "w") do io
        println(io, "x y")
        writedlm(io, [Ecuts iters], ' ')
    end
    open("experiments/data/ecuts_$(name)_perc.dat", "w") do io
        println(io, "x y")
        writedlm(io, [Ecuts perc], ' ')
    end
end
display(plt1)
display(plt2)

plt1 = plot(; ylabel = "hams", xlabel = "Ecut")
plt2 = plot(; ylabel = "perc", xlabel = "Ecut")
for (cbs, name) = zip(cbss, ["EARCG-Gr", "EARCG-St", "H1RCG", "L2RCG", "SCF"])
    hamss = get_iters(cbs; hams = true)
    perc = hamss./hamss[1] 
    plot!(plt1, Ecuts, hamss, label = name)
    plot!(plt2, Ecuts, perc, label = name)
    open("experiments/data/ecuts_$(name)_hams.dat", "w") do io
        println(io, "x y")
        writedlm(io, [Ecuts hamss], ' ')
    end
    open("experiments/data/ecuts_$(name)_hamperc.dat", "w") do io
        println(io, "x y")
        writedlm(io, [Ecuts perc], ' ')
    end
end
display(plt1)
display(plt2)
