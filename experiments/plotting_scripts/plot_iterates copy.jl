function get_rho(ρ, lattice; n_pix = 400)

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

    dataX = range(- 1.1 * a, stop= 1.1 * a, length = n_pix)
    dataY = range(- 1.1 * a, stop= 1.1 * a, length = n_pix)
    z = 0.125 * a

    return [get_val(dataX[i], dataY[j], z, ρ[:,:,:,1]) for i = 1:n_pix, j = 1:n_pix]
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

function plot_energy_distribution(ψ, basis, Ecuts; ylims = (1e-13,2))
    as, es = calculate_energy_distribution(ψ, basis)


end


struct PlotDiffCallback
    name 
    ψ_ref
    ρ_ref
    basis_ref
    Ecuts
    default_callback
end

# heatmap with a log10 colorbar; cticks sets the colorbar ticks, clims the color limits (in units of A,
# default: extrema of A), colorbar = false omits the colorbar, other kwargs go to the main heatmap
function logheatmap(A; cticks = nothing, clims = nothing, colorbar = true, color = :inferno, dpi = 100, kw...)
    lo, hi = log10.(something(clims, extrema(A)))
    cticks = something(cticks, round.(exp10.(range(lo, hi, 5)), sigdigits = 2))
    cvals = range(lo, hi, 256)
    main = heatmap(log10.(A); clims = (lo, hi), color, colorbar = false, grid = false, kw...)
    colorbar || return plot(main; dpi)
    cbar = heatmap([0, 1], cvals, repeat(cvals, 1, 2); clims = (lo, hi), color, colorbar = false,
                   grid = false, xticks = false, ymirror = true,
                   yticks = (log10.(cticks), string.(cticks)), ylims = (lo, hi), framestyle = :box)
    plot(main, cbar; layout = grid(1, 2, widths = [0.93, 0.07]), dpi)
end

function (cb::PlotDiffCallback)(info)
    (info.stage == :finalize) && return cb.default_callback(info)

    title = "$(cb.name)-$(info.n_iter)"
    img_path = "./experiments/imgs/"
    # option A: extrapolate ρ
    # option B: recalculate density

    ψ_iter = RCG_DFTK.interpolate_c2f(info.basis, cb.basis_ref, info.ψ)

    ρ_iter = RCG_DFTK.compute_density(cb.basis_ref, ψ_iter)

    rho = get_rho(abs.(ρ_iter .- cb.ρ_ref), cb.basis_ref.model.lattice)

    title = "$title-error"   
    hmp = logheatmap(rho ; color=:magma, dpi = 300, title="test", 
        xaxis=false, yaxis=false, aspect_ratio = :equal, #colorbar=:none,
        showaxis = false, ticks= false)

    savefig(hmp, "$img_path$(title).png")
    display(hmp)

    cb.default_callback(info)
end
