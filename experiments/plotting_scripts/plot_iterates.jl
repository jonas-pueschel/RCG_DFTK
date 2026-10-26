function get_rho(ρ, lattice; n_pix = 300)

    function eval_cube(vals, coords)
        vals1 = coords[1] * vals[2, :, :] + (1- coords[1]) * vals[1, :, :]
        vals2 = coords[2] * vals1[2,  :] + (1- coords[2]) * vals1[1, :]
        return coords[3] * vals2[2] + (1- coords[3]) * vals2[1]
    end

    function get_val(x,y,z, values_raw)
        #get lattice coordinates

        xl, yl, zl = lattice\[x,y,z]

        #unit cell
        coords_fl = xl%1, yl%1, zl%1
        coords_fl = [c >= 0 ? c : c + 1 for c = coords_fl]


        coords_floor = [mod(Int(floor(coords_fl[i] * (size(values_raw)[i]))),size(values_raw)[i]) + 1 for i = 1:3]
        coords_ceil = [mod(Int(ceil(coords_fl[i] * (size(values_raw)[i]))),size(values_raw)[i]) + 1 for i = 1:3]
        coords_offset = [mod(coords_fl[i] * (size(values_raw)[i]),size(values_raw)[i]) + 1 for i = 1:3] - coords_floor

        # 8 -> 4 -> 2 -> 1
        vals = zeros(2,2,2)
        for i=1:2, j = 1:2, k = 1:2
            ijk = [i, j, k]
            coords = [ijk[l] == 1 ? coords_floor[l] : coords_ceil[l] for l = 1:3]
            vals[i,j,k] = values_raw[coords...]
        end

        return eval_cube(vals, coords_offset)    
    end

    a = max(model.lattice...)

    dataX = range(-0.7 * a, stop= 0.5 * a, length = n_pix)
    dataY = range(- 0.7 * a, stop= 0.5 * a, length = n_pix)
    z = 0.125 * a

    return [get_val(dataX[i], dataY[j], z, ρ[:,:,:,1]) for i = 1:lgth, j = 1:lgth]
end

function calculate_energy_distribution(ψ, basis)
    amplitudes = []
    e_kins = []

    for (kpt, ψk) = zip(basis.kpoints, ψ)

        Pk = PreconditionerTPA(basis, kpt)
        for i = 1:size(ψk)[1]
            push!(amplitudes, sqrt(sum(abs(ψk[i, j])^2 for j = 1:size(ψk)[2])))
            #probably one can push the whole Pk.kin at once
            push!(e_kins, Pk.kin[i])
        end
    end
    return amplitudes, e_kins
end

as, es = calculate_energy_distribution(scfres_ref.ψ, scfres_ref.basis)
xs = sqrt.(2 .* es)          # |k+G|
Eticks = [0, 25, 100, 200, 400, 630]
scatter(xs, as, yscale = :log10,
        xticks = (sqrt.(2 .* Eticks), string.(Eticks)),
        xlabel = "E_kin (√ scale)", ylims = (1e-13, 2))


struct PlotDiffCallback
    name 
    ψ_ref
    ρ_ref
    basis_ref
    default_callback
end

function (cb::PlotDiffCallback)(info)

    # if (info.converged)
    #     rho = info.ρ
    #     title = "$name-final"
    # else
    if (#info.n_iter in print_on_iter && 
        !info.converged)
        title = "$(cb.name)-$(info.n_iter)"
        rho = info.ρout
    else
        return default_callback(info)
    end

    rhoshift = get_rho(rho)


    hmp = heatmap(rhoshift; dpi = 300, title=title, 
            xaxis=false, yaxis=false, aspect_ratio = :equal, colorbar=:none,
            showaxis = false, ticks= false,
            clims = (0,rho_max))

    savefig(hmp, "$img_path$(title).png")
    display(hmp)

    title = "$title-error"

    rhoerr = log.(abs.(rhoshift - rho_ref))

    errmax = 0.0001
    errmin = 1e-12

    #=     if (haskey(info, :coarse_corrections) && info.coarse_corrections[end])
        hmp = heatmap(rhoerr;  dpi = 300, 
        # background_color_outside=colorant"#39FF14",
        # background_color_inside=colorant"#39FF14", 
        title=title, 
            xaxis=false, yaxis=false, aspect_ratio = :equal, colorbar=:none,
            showaxis = false, ticks= false, clims = (log(errmin), log(errmax)))
    else =#
    hmp = heatmap(rhoerr; color=:magma, dpi = 300, title=title, 
        xaxis=false, yaxis=false, aspect_ratio = :equal, #colorbar=:none,
        showaxis = false, ticks= false)
    #=     end =#

    savefig(hmp, "$img_path$(title).png")
    display(hmp)





    default_callback(info)
end



# name = "H1-7L"
# println("\n$name")

# result = multilevel_riemannian_optimization(basis_arr; ψ = ψ1, ρ = ρ1, tol,
#     callback = cb,
#     coarse_model_tol = 1e-2,
#     coarse_cond_tol = 0.45,
#     gradients = [H1Gradient(basis) for basis = basis_arr]);


name = "H1-4L"
println("\n$name")
result = multilevel_riemannian_optimization(basis_arr[[1,3,5,7]]; ψ = ψ1, ρ = ρ1, tol,
    callback = cb,
    coarse_model_tol = 1e-2,
    coarse_cond_tol = 0.45,
    gradients = [H1Gradient(basis) for basis = basis_arr[[1,3,5,7]]]);

# name = "H1-3L"
# println("\n$name")
# result = multilevel_riemannian_optimization(basis_arr[[1,4,7]]; ψ = ψ1, ρ = ρ1, tol,
#     callback = cb,
#     coarse_model_tol = 1e-2,
#     coarse_cond_tol = 0.45,
#     gradients = [H1Gradient(basis) for basis = basis_arr[[1,4,7]]]);

# name = "H1-2L"
# println("\n$name")
# result = two_level_riemannian_optimization(basis_c, basis_f; ψ = ψ1, ρ = ρ1, tol,
#     coarse_model_tol = 1e-2,
#     coarse_cond_tol = 0.45,
#     iteration_strat_fine = StandardBacktracking(
#         ArmijoRule(0.1, 0.5),
#         ConstantStep(1.0), 10
#     ),
#     callback = cb,
#     gradient = H1Gradient(basis_f),
#     gradient_c = H1Gradient(basis_c),
#     );

name = "H1RCG"
println("\n$name")
result = RCG_DFTK.riemannian_conjugate_gradient(basis_f;
    callback = cb, ψ = ψ1, ρ = ρ1, tol, gradient = H1Gradient(basis_f));


# name = "H1RG"
# println("\n$name")
# result = RCG_DFTK.riemannian_conjugate_gradient(basis_f;
#     iteration_strat = StandardBacktracking(NonmonotoneRule(0.95, 0.05, 0.5), BarzilaiBorweinStep(0.1, 2.5, ConstantStep(1.0)), 10),
#     cg_param = ParamZero(),
#     callback = cb, ψ = ψ1, ρ = ρ1, tol, gradient = H1Gradient(basis_f));